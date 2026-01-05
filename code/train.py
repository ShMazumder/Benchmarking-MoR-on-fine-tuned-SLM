"""
Training script for MoR benchmarking experiments
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os

from config import Config
from data import get_shakespeare_loaders, get_wikitext_loaders
from models import BaselineTransformer, MoRTransformer
from utils import calculate_accuracy, save_checkpoint, save_results, Timer, print_model_info

def train_baseline(model, train_loader, test_loader, config, experiment_name):
    """
    Train baseline Transformer model
    
    Args:
        model: BaselineTransformer instance
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration object
        experiment_name: Name for saving results
    
    Returns:
        results: Dictionary with training results
    """
    device = config.device
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Timer
    timer = Timer()
    
    # Training loop
    print(f"\nTraining {experiment_name}...")
    timer.start()
    
    for epoch in range(config.epochs_baseline):
        model.train()
        total_loss = 0
        total_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs_baseline}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, effective_depth = model(x)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            acc = calculate_accuracy(logits, y)
            
            total_loss += loss.item()
            total_acc += acc
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'depth': f'{effective_depth:.2f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
    
    timer.stop()
    training_time = timer.get_elapsed()
    
    # Evaluation
    print("\nEvaluating...")
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            test_acc += calculate_accuracy(logits, y)
    
    test_acc /= len(test_loader)
    
    # Results
    results = {
        'experiment': experiment_name,
        'model_type': 'baseline',
        'n_layers': model.n_layers,
        'accuracy': avg_acc,
        'test_accuracy': test_acc,
        'effective_depth': float(model.n_layers),
        'training_time_seconds': training_time
    }
    
    print(f"\nResults:")
    print(f"  Training Accuracy: {avg_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Effective Depth: {model.n_layers}")
    print(f"  Training Time: {training_time:.0f}s")
    
    return results

def train_mor(model, train_loader, test_loader, config, experiment_name, epochs, lambda_penalty=0.1):
    """
    Train MoR Transformer model
    
    Args:
        model: MoRTransformer instance
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration object
        experiment_name: Name for saving results
        epochs: Number of training epochs
        lambda_penalty: Weight for depth penalty
    
    Returns:
        results: Dictionary with training results
    """
    device = config.device
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Timer
    timer = Timer()
    
    # Training loop
    print(f"\nTraining {experiment_name}...")
    timer.start()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_depth = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, effective_depth, routing_stats = model(x, training=True)
            
            # Calculate loss with depth penalty
            ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            depth_penalty = lambda_penalty * effective_depth
            loss = ce_loss + depth_penalty
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            acc = calculate_accuracy(logits, y)
            
            total_loss += ce_loss.item()
            total_acc += acc
            total_depth += effective_depth.item()
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'depth': f'{effective_depth.item():.2f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_depth = total_depth / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, Depth={avg_depth:.2f}")
    
    timer.stop()
    training_time = timer.get_elapsed()
    
    # Evaluation
    print("\nEvaluating...")
    model.eval()
    test_acc = 0
    test_depth = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, effective_depth, routing_stats = model(x, training=False)
            test_acc += calculate_accuracy(logits, y)
            test_depth += effective_depth.item()
    
    test_acc /= len(test_loader)
    test_depth /= len(test_loader)
    
    # Results
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
    
    print(f"\nResults:")
    print(f"  Training Accuracy: {avg_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Effective Depth: {avg_depth:.2f}")
    print(f"  Test Effective Depth: {test_depth:.2f}")
    print(f"  Training Time: {training_time:.0f}s")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train MoR Benchmarking Models')
    parser.add_argument('--dataset', type=str, default='shakespeare', choices=['shakespeare', 'wikitext'],
                        help='Dataset to use')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['baseline_6', 'baseline_12', 'mor_exp1', 'mor_exp2'],
                        help='Which experiment to run')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()
    
    config = Config()
    config.device = args.device
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'shakespeare':
        train_loader, test_loader, vocab_size = get_shakespeare_loaders(
            batch_size=config.batch_size,
            seq_length=config.max_seq_len
        )
    else:
        train_loader, val_loader, test_loader, vocab_size = get_wikitext_loaders(
            batch_size=config.batch_size,
            seq_length=config.max_seq_len
        )
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Run experiment
    if args.experiment == 'baseline_6':
        model = BaselineTransformer(vocab_size, n_layers=6, **vars(config))
        print_model_info(model, "Baseline Transformer (N=6)")
        results = train_baseline(model, train_loader, test_loader, config, "Baseline_N6")
        
    elif args.experiment == 'baseline_12':
        model = BaselineTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "Baseline Transformer (N=12)")
        results = train_baseline(model, train_loader, test_loader, config, "Baseline_N12")
        
    elif args.experiment == 'mor_exp1':
        model = MoRTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "MoR Transformer (Exp 1)")
        results = train_mor(model, train_loader, test_loader, config, "MoR_Exp1",
                           epochs=config.epochs_mor_exp1, lambda_penalty=0.1)
        
    elif args.experiment == 'mor_exp2':
        model = MoRTransformer(vocab_size, n_layers=12, **vars(config))
        print_model_info(model, "MoR Transformer (Exp 2)")
        results = train_mor(model, train_loader, test_loader, config, "MoR_Exp2",
                           epochs=config.epochs_mor_exp2, lambda_penalty=0.05)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_results(results, f'results/{args.dataset}_{args.experiment}.json')

if __name__ == '__main__':
    main()
