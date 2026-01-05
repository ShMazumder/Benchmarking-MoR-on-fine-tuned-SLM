"""
Tiny Shakespeare Dataset Loader
Character-level tokenization
"""
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os

class TinyShakespeareDataset(Dataset):
    def __init__(self, seq_length=64, split='train', split_ratio=0.9):
        """
        Args:
            seq_length: Length of each sequence
            split: 'train' or 'test'
            split_ratio: Ratio for train/test split
        """
        self.seq_length = seq_length
        
        # Download or load data
        data_path = 'data/shakespeare.txt'
        if not os.path.exists(data_path):
            os.makedirs('data', exist_ok=True)
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            response = requests.get(url)
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode entire text
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        
        # Split data
        split_idx = int(len(data) * split_ratio)
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence [seq_length]
            y: Target sequence [seq_length] (shifted by 1)
        """
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y

def get_shakespeare_loaders(batch_size=64, seq_length=64, split_ratio=0.9):
    """
    Get train and test dataloaders for Tiny Shakespeare
    """
    train_dataset = TinyShakespeareDataset(seq_length, 'train', split_ratio)
    test_dataset = TinyShakespeareDataset(seq_length, 'test', split_ratio)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.vocab_size

if __name__ == '__main__':
    # Test the dataset
    train_loader, test_loader, vocab_size = get_shakespeare_loaders()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, {y.shape}")
