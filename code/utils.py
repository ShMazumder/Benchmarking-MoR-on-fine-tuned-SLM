"""
Utility functions for training and evaluation
"""
import torch
import time
import json
import os

def calculate_accuracy(logits, targets):
    """
    Calculate top-1 accuracy for next-token prediction
    
    Args:
        logits: Model output [batch, seq_len, vocab_size]
        targets: Target tokens [batch, seq_len]
    
    Returns:
        accuracy: Percentage of correct predictions
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return 100.0 * correct / total

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def save_results(results, filepath):
    """Save experimental results to JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {filepath}")

class Timer:
    """Simple timer for measuring training time"""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.start_time = None
    
    def reset(self):
        self.start_time = None
        self.elapsed = 0
    
    def get_elapsed(self):
        return self.elapsed

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, name="Model"):
    """Print model information"""
    num_params = count_parameters(model)
    print(f"\n{name} Information:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: {num_params * 4 / 1024 / 1024:.2f} MB (FP32)")
