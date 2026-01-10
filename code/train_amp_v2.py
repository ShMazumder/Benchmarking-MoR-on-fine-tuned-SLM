"""
AGGRESSIVE TRAINING FIXES - train_amp_v2.py
This version implements:
1. Learning rate warmup + cosine annealing
2. Aggressive gradient clipping
3. Lower base learning rate (1e-4)
4. Smaller batch size option
5. Better optimizer settings
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import json
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import math

from config import Config
from data import get_shakespeare_loaders, get_wikitext_loaders, get_bangla_loaders
from models import BaselineTransformer, MoRTransformer
from utils import calculate_accuracy, save_checkpoint, save_results, Timer, print_model_info
from torch.amp import autocast, GradScaler


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    Create a schedule with linear warmup and cosine annealing.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_move_model(model, device):
    from torch.nn.parallel import DistributedDataParallel as _DDP
    if isinstance(model, _DDP):
        return model
    return model.to(device)


def train_baseline(model, train_loader, test_loader, config, experiment_name):
    device = config.device
    model = maybe_move_model(model, device)

    is_main = not dist.is_initialized() or dist.get_rank() == 0

    # AGGRESSIVE FIX 1: Lower learning rate with better optimizer settings
    base_lr = 1e-4  # Reduced from 3e-4
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01  # Add weight decay for regularization
    )
    criterion = nn.CrossEntropyLoss()

    # AGGRESSIVE FIX 2: Learning rate scheduler with warmup
    total_steps = len(train_loader) * config.epochs_baseline
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = True if str(config.device).startswith('cuda') else False
    scaler = GradScaler('cuda') if use_amp else None

    timer = Timer()
    history = []

    print(f"\\nTraining {experiment_name} with AGGRESSIVE OPTIMIZATIONS...")
    print(f"  Base LR: {base_lr}, Warmup steps: {warmup_steps}, Total steps: {total_steps}")
    print(f"  Gradient clipping: 1.0, Weight decay: 0.01")
    timer.start()

    global_step = 0
    for epoch in range(config.epochs_baseline):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs_baseline}", disable=(not is_main))
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            if use_amp:
                with autocast('cuda'):
                    logits, effective_depth = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # AGGRESSIVE FIX 3: Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, effective_depth = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                
                # AGGRESSIVE FIX 3: Gradient clipping (CPU)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

            # AGGRESSIVE FIX 2: Step scheduler
            scheduler.step()
            global_step += 1

            acc = calculate_accuracy(logits, y)
            total_loss += float(loss.item())
            total_acc += float(acc)

            # Show current LR in progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{acc:.2f}%', 
                'depth': f'{float(effective_depth):.2f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, LR={current_lr:.2e}")

        # Checkpoint (only main process)
        if is_main:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch+1, 
                'model_state': state_dict, 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, Path(config.checkpoint_dir)/f"{experiment_name}_epoch{epoch+1}.pt")

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': avg_acc, 'lr': current_lr})

    timer.stop()
    training_time = timer.get_elapsed()

    # Evaluation
    if is_main:
        print("\\nEvaluating...")
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            test_loss += float(loss.item())
            test_acc += calculate_accuracy(logits, y)
    
    test_acc /= len(test_loader)
    test_loss /= len(test_loader)

    if is_main:
        results_path = Path(config.results_dir) / f"{experiment_name}.json"
        save_results({
            'training_accuracy': avg_acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'effective_depth': 'N/A',
            'training_time': training_time
        }, results_path)
        print(f"Results saved: {results_path}")
        print(f"\\nResults:\\n  Training Accuracy: {avg_acc:.2f}%\\n  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}\\n  Training Time: {training_time}s")
        
        # Save training history
        history_path = Path(config.results_dir) / f"{experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    return avg_acc, test_acc, training_time


def train_mor(model, train_loader, test_loader, config, experiment_name, lambda_aux=0.1):
    """MoR training with same aggressive optimizations"""
    device = config.device
    model = maybe_move_model(model, device)
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    # Same aggressive settings
    base_lr = 1e-4
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * config.epochs_mor
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = True if str(config.device).startswith('cuda') else False
    scaler = GradScaler('cuda') if use_amp else None

    timer = Timer()
    history = []

    print(f"\\nTraining {experiment_name} with AGGRESSIVE OPTIMIZATIONS...")
    print(f"  Base LR: {base_lr}, Warmup steps: {warmup_steps}, Lambda: {lambda_aux}")
    timer.start()

    global_step = 0
    for epoch in range(config.epochs_mor):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_depth = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs_mor}", disable=(not is_main))
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            if use_amp:
                with autocast('cuda'):
                    logits, effective_depth, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = ce_loss + lambda_aux * aux_loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, effective_depth, aux_loss = model(x, return_aux_loss=True)
                ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = ce_loss + lambda_aux * aux_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1

            acc = calculate_accuracy(logits, y)
            total_loss += float(ce_loss.item())
            total_acc += float(acc)
            total_depth += float(effective_depth)

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.4f}', 
                'acc': f'{acc:.2f}%', 
                'depth': f'{float(effective_depth):.2f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_depth = total_depth / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, Depth={avg_depth:.2f}, LR={current_lr:.2e}")

        if is_main:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch+1,
                'model_state': state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, Path(config.checkpoint_dir)/f"{experiment_name}_epoch{epoch+1}.pt")

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': avg_acc, 'depth': avg_depth, 'lr': current_lr})

    timer.stop()
    training_time = timer.get_elapsed()

    # Evaluation
    if is_main:
        print("\\nEvaluating...")
    model.eval()
    test_acc = 0.0
    test_depth = 0.0
    test_loss = 0.0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, eff_depth = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            test_loss += float(loss.item())
            test_acc += calculate_accuracy(logits, y)
            test_depth += float(eff_depth)
    
    test_acc /= len(test_loader)
    test_depth /= len(test_loader)
    test_loss /= len(test_loader)

    if is_main:
        results_path = Path(config.results_dir) / f"{experiment_name}.json"
        save_results(results_path, {
            'training_accuracy': avg_acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'effective_depth': avg_depth,
            'test_effective_depth': test_depth,
            'training_time': training_time
        })
        print(f"Results saved: {results_path}")
        print(f"\\nResults:\\n  Training Accuracy: {avg_acc:.2f}%\\n  Test Accuracy: {test_acc:.2f}%")
        print(f"  Effective Depth: {avg_depth:.2f}\\n  Test Effective Depth: {test_depth:.2f}")
        print(f"  Test Loss: {test_loss:.4f}\\n  Training Time: {training_time}s")
        
        history_path = Path(config.results_dir) / f"{experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    return avg_acc, test_acc, avg_depth, test_depth, training_time


    return avg_acc, test_acc, avg_depth, test_depth, training_time


def main():
    parser = argparse.ArgumentParser(description='Train MoR Transformer with Aggressive Optimizations')
    parser.add_argument('--dataset', type=str, default='shakespeare', choices=['shakespeare', 'wikitext', 'bangla'])
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['baseline_6', 'baseline_12', 'mor_exp1', 'mor_exp2'])
    parser.add_argument('--tokenization', type=str, default='char', choices=['char', 'subword'])
    parser.add_argument('--tokenizer_model', type=str, default=None)
    parser.add_argument('--subword_vocab_size', type=int, default=8000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--syncbn', action='store_true', help='Use SyncBatchNorm for distributed')
    args = parser.parse_args()

    # Distributed setup
    is_distributed = args.local_rank != -1
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = Config()
    config.device = device

    # Override epochs if specified
    if args.epochs:
        config.epochs_baseline = args.epochs
        config.epochs_mor = args.epochs

    # Load dataset
    if args.dataset == 'shakespeare':
        train_loader, test_loader, vocab_size = get_shakespeare_loaders(
            batch_size=config.batch_size,
            seq_length=config.max_seq_len,
            tokenization=args.tokenization,
            tokenizer_model=args.tokenizer_model,
            vocab_size=args.subword_vocab_size
        )
    elif args.dataset == 'wikitext':
        train_loader, val_loader, test_loader, vocab_size = get_wikitext_loaders(
            batch_size=config.batch_size,
            seq_length=config.max_seq_len,
            tokenization=args.tokenization,
            tokenizer_model=args.tokenizer_model,
            vocab_size=args.subword_vocab_size
        )
    elif args.dataset == 'bangla':
        train_loader, test_loader, vocab_size = get_bangla_loaders(
            batch_size=config.batch_size,
            seq_length=config.max_seq_len,
            tokenization=args.tokenization,
            tokenizer_model=args.tokenizer_model,
            vocab_size=args.subword_vocab_size
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # Distributed sampler
    if is_distributed:
        train_dataset = train_loader.dataset
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler,
                                  num_workers=2, pin_memory=True)
        test_dataset = test_loader.dataset
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

    print(f"Vocabulary size: {vocab_size}")

    # Run experiment with aggressive training
    experiment_name = f"{args.dataset}_{args.experiment}"
    
    if args.experiment == 'baseline_6':
        model = BaselineTransformer(vocab_size, n_layers=6, **vars(config))
        print_model_info(model, "Baseline Transformer (N=6)")
        if is_distributed:
            model.to(device)
            model = DDP(model, device_ids=[args.local_rank] if str(device).startswith('cuda') else None)
        train_baseline(model, train_loader, test_loader, config, experiment_name)

    elif args.experiment == 'baseline_12':
        model = BaselineTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "Baseline Transformer (N=12)")
        if is_distributed:
            model.to(device)
            model = DDP(model, device_ids=[args.local_rank] if str(device).startswith('cuda') else None)
        train_baseline(model, train_loader, test_loader, config, experiment_name)

    elif args.experiment == 'mor_exp1':
        model = MoRTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "MoR Transformer (Exp 1)")
        if is_distributed:
            model.to(device)
            model = DDP(model, device_ids=[args.local_rank] if str(device).startswith('cuda') else None)
        train_mor(model, train_loader, test_loader, config, experiment_name, lambda_aux=0.1)

    elif args.experiment == 'mor_exp2':
        model = MoRTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "MoR Transformer (Exp 2)")
        if is_distributed:
            model.to(device)
            model = DDP(model, device_ids=[args.local_rank] if str(device).startswith('cuda') else None)
        train_mor(model, train_loader, test_loader, config, experiment_name, lambda_aux=0.05)


if __name__ == '__main__':
    main()
