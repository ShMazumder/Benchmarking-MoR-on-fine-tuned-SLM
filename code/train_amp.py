"""
AMP-enhanced training script (quick test)
This is a safe, standalone copy of `train.py` with AMP (mixed precision),
checkpointing, and per-epoch history writing for quick experiments.
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

from config import Config
from data import get_shakespeare_loaders, get_wikitext_loaders, get_bangla_loaders
from models import BaselineTransformer, MoRTransformer
from utils import calculate_accuracy, save_checkpoint, save_results, Timer, print_model_info
from torch.amp import autocast, GradScaler


def maybe_move_model(model, device):
    from torch.nn.parallel import DistributedDataParallel as _DDP
    if isinstance(model, _DDP):
        return model
    return model.to(device)


def train_baseline(model, train_loader, test_loader, config, experiment_name):
    device = config.device
    model = maybe_move_model(model, device)

    is_main = not dist.is_initialized() or dist.get_rank() == 0

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    use_amp = True if str(config.device).startswith('cuda') else False
    scaler = GradScaler('cuda') if use_amp else None

    timer = Timer()
    history = []

    print("\nTraining {}...".format(experiment_name))
    timer.start()

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
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, effective_depth = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = calculate_accuracy(logits, y)
            total_loss += float(loss.item())
            total_acc += float(acc)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%', 'depth': f'{float(effective_depth):.2f}'})

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")

        # checkpoint (only main process)
        if is_main:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({'epoch': epoch+1, 'model_state': state_dict, 'optimizer': optimizer.state_dict()}, Path(config.checkpoint_dir)/f"{experiment_name}_epoch{epoch+1}.pt")

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': avg_acc})

    timer.stop()
    training_time = timer.get_elapsed()

    # Evaluation (main prints only)
    if is_main:
        print("\nEvaluating...")
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            test_acc += float(calculate_accuracy(logits, y))

    test_acc /= len(test_loader)

    results = {
        'experiment': experiment_name,
        'model_type': 'baseline',
        'n_layers': model.n_layers,
        'accuracy': avg_acc,
        'test_accuracy': test_acc,
        'effective_depth': float(model.n_layers),
        'training_time_seconds': training_time
    }

    if is_main:
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        save_results(results, f"{config.results_dir}/{experiment_name}.json")
        with open(f"{config.results_dir}/{experiment_name}_history.json", 'w') as fh:
            json.dump(history, fh)

        print("\nResults:")
        print("  Training Accuracy: {:.2f}%".format(avg_acc))
        print("  Test Accuracy: {:.2f}%".format(test_acc))
        print("  Effective Depth: {}".format(model.n_layers))
        print("  Training Time: {:.0f}s".format(training_time))

    return results


def train_baseline_amp(model, train_loader, test_loader, config, experiment_name):
    """AMP-specific wrapper that preserves original `train_baseline`.
    This avoids overriding the original function symbol and keeps both variants available.
    """
    return train_baseline(model, train_loader, test_loader, config, experiment_name)


def train_mor_amp(model, train_loader, test_loader, config, experiment_name, epochs, lambda_penalty=0.1):
    """AMP-specific wrapper that preserves original `train_mor`.
    """
    return train_mor(model, train_loader, test_loader, config, experiment_name, epochs, lambda_penalty)


def train_mor(model, train_loader, test_loader, config, experiment_name, epochs, lambda_penalty=0.1):
    device = config.device
    model = maybe_move_model(model, device)

    is_main = not dist.is_initialized() or dist.get_rank() == 0

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    use_amp = True if str(config.device).startswith('cuda') else False
    scaler = GradScaler('cuda') if use_amp else None

    timer = Timer()
    history = []

    print("\nTraining {}...".format(experiment_name))
    timer.start()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_depth = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(not is_main))
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            if use_amp:
                with autocast('cuda'):
                    logits, effective_depth, routing_stats = model(x, training=True)
                    ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    depth_penalty = lambda_penalty * effective_depth
                    loss = ce_loss + depth_penalty
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, effective_depth, routing_stats = model(x, training=True)
                ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                depth_penalty = lambda_penalty * effective_depth
                loss = ce_loss + depth_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = calculate_accuracy(logits, y)
            total_loss += float(ce_loss.item())
            total_acc += float(acc)
            total_depth += float(effective_depth)

            pbar.set_postfix({'loss': f'{ce_loss.item():.4f}', 'acc': f'{acc:.2f}%', 'depth': f'{float(effective_depth):.2f}'})

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_depth = total_depth / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, Depth={avg_depth:.2f}")

        if is_main:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({'epoch': epoch+1, 'model_state': state_dict, 'optimizer': optimizer.state_dict()}, Path(config.checkpoint_dir)/f"{experiment_name}_epoch{epoch+1}.pt")
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': avg_acc, 'depth': avg_depth})

    timer.stop()
    training_time = timer.get_elapsed()

    if is_main:
        print("\nEvaluating...")
    model.eval()
    test_acc = 0.0
    test_depth = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, effective_depth, routing_stats = model(x, training=False)
            test_acc += float(calculate_accuracy(logits, y))
            test_depth += float(effective_depth)

    test_acc /= len(test_loader)
    test_depth /= len(test_loader)

    results = {
        'experiment': experiment_name,
        'model_type': 'mor',
        'n_layers': model.n_layers,
        'accuracy': avg_acc,
        'test_accuracy': test_acc,
        'effective_depth': avg_depth,
        'test_effective_depth': test_depth,
        'training_time_seconds': training_time,
        'lambda_penalty': lambda_penalty
    }

    if is_main:
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        save_results(results, f"{config.results_dir}/{experiment_name}.json")
        with open(f"{config.results_dir}/{experiment_name}_history.json", 'w') as fh:
            json.dump(history, fh)

        print("\nResults:")
        print("  Training Accuracy: {:.2f}%".format(avg_acc))
        print("  Test Accuracy: {:.2f}%".format(test_acc))
        print("  Effective Depth: {:.2f}".format(avg_depth))
        print("  Test Effective Depth: {:.2f}".format(test_depth))
        print("  Training Time: {:.0f}s".format(training_time))

    return results


def main():
    parser = argparse.ArgumentParser(description='Train MoR Benchmarking Models')
    parser.add_argument('--dataset', type=str, default='shakespeare', choices=['shakespeare', 'wikitext', 'bangla'],
                        help='Dataset to use')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['baseline_6', 'baseline_12', 'mor_exp1', 'mor_exp2'],
                        help='Which experiment to run')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--amp', action='store_true', help='Use AMP wrapper variants')
    parser.add_argument('--syncbn', action='store_true', help='Convert BatchNorm to SyncBatchNorm before DDP')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)),
                        help='Local rank for distributed training')
    parser.add_argument('--tokenization', type=str, choices=['char', 'word', 'subword'], default=None,
                        help='Tokenization to use (char, word, subword). If omitted, uses config default')
    parser.add_argument('--subword_vocab_size', type=int, default=None,
                        help='Vocabulary size when using subword tokenization')
    parser.add_argument('--tokenizer_model', type=str, default=None,
                        help='Path to a pre-trained SentencePiece model to use for subword tokenization')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs for quick smoke runs')
    args = parser.parse_args()

    config = Config()
    # allow overriding epochs for quick tests
    if args.epochs is not None:
        config.epochs_baseline = args.epochs
        config.epochs_mor_exp1 = args.epochs
        config.epochs_mor_exp2 = args.epochs
    # Distributed setup
    is_distributed = args.local_rank != -1
    if is_distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device(args.device)
    config.device = device

    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'shakespeare':
        train_loader, test_loader, vocab_size = get_shakespeare_loaders(
            batch_size=config.batch_size,
            seq_length=config.max_seq_len,
            tokenization=args.tokenization,
            tokenizer_model=args.tokenizer_model,
            vocab_size=args.subword_vocab_size
        )
        val_loader = None
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
        val_loader = None
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # If distributed, replace train_loader (and optionally test_loader) with ones using DistributedSampler
    if is_distributed:
        # Recreate loaders with DistributedSampler to partition data across processes
        train_dataset = train_loader.dataset
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler,
                                  num_workers=getattr(train_loader, 'num_workers', 2), pin_memory=True)
        # For evaluation, keep deterministic ordering
        test_dataset = test_loader.dataset
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=getattr(test_loader, 'num_workers', 2), pin_memory=True)

    print(f"Vocabulary size: {vocab_size}")

    # Run experiment
    if args.experiment == 'baseline_6':
        model = BaselineTransformer(vocab_size, n_layers=6, **vars(config))
        print_model_info(model, "Baseline Transformer (N=6)")
        # Optionally convert BatchNorm to SyncBatchNorm before DDP
        if is_distributed and args.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Wrap model with DDP if running distributed
        if is_distributed:
            model.to(device)
            if str(device).startswith('cuda'):
                model = DDP(model, device_ids=[args.local_rank])
            else:
                model = DDP(model)
        if args.amp:
            results = train_baseline_amp(model, train_loader, test_loader, config, "Baseline_N6")
        else:
            results = train_baseline(model, train_loader, test_loader, config, "Baseline_N6")

    elif args.experiment == 'baseline_12':
        model = BaselineTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "Baseline Transformer (N=12)")
        if is_distributed and args.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_distributed:
            model.to(device)
            if str(device).startswith('cuda'):
                model = DDP(model, device_ids=[args.local_rank])
            else:
                model = DDP(model)
        if args.amp:
            results = train_baseline_amp(model, train_loader, test_loader, config, "Baseline_N12")
        else:
            results = train_baseline(model, train_loader, test_loader, config, "Baseline_N12")

    elif args.experiment == 'mor_exp1':
        model = MoRTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "MoR Transformer (Exp 1)")
        if is_distributed and args.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_distributed:
            model.to(device)
            if str(device).startswith('cuda'):
                model = DDP(model, device_ids=[args.local_rank])
            else:
                model = DDP(model)
        if args.amp:
            results = train_mor_amp(model, train_loader, test_loader, config, "MoR_Exp1",
                                   epochs=config.epochs_mor_exp1, lambda_penalty=0.1)
        else:
            results = train_mor(model, train_loader, test_loader, config, "MoR_Exp1",
                           epochs=config.epochs_mor_exp1, lambda_penalty=0.1)

    elif args.experiment == 'mor_exp2':
        model = MoRTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "MoR Transformer (Exp 2)")
        if is_distributed and args.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_distributed:
            model.to(device)
            if str(device).startswith('cuda'):
                model = DDP(model, device_ids=[args.local_rank])
            else:
                model = DDP(model)
        if args.amp:
            results = train_mor_amp(model, train_loader, test_loader, config, "MoR_Exp2",
                                   epochs=config.epochs_mor_exp2, lambda_penalty=0.05)
        else:
            results = train_mor(model, train_loader, test_loader, config, "MoR_Exp2",
                           epochs=config.epochs_mor_exp2, lambda_penalty=0.05)

    # Save results
    os.makedirs('results', exist_ok=True)
    save_results(results, f'results/{args.dataset}_{args.experiment}.json')


if __name__ == '__main__':
    main()
