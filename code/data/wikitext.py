"""
WikiText-2 Dataset Loader
Character-level tokenization (to match Shakespeare experiments)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os

class WikiText2Dataset(Dataset):
    def __init__(self, seq_length=64, split='train'):
        """
        Args:
            seq_length: Length of each sequence
            split: 'train', 'validation', or 'test'
        """
        self.seq_length = seq_length
        
        # Load WikiText-2 from HuggingFace
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Concatenate all text
        text = ' '.join(dataset['text'])
        
        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode entire text
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
    
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

def get_wikitext_loaders(batch_size=64, seq_length=64):
    """
    Get train, validation, and test dataloaders for WikiText-2
    """
    train_dataset = WikiText2Dataset(seq_length, 'train')
    val_dataset = WikiText2Dataset(seq_length, 'validation')
    test_dataset = WikiText2Dataset(seq_length, 'test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
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
    
    return train_loader, val_loader, test_loader, train_dataset.vocab_size

if __name__ == '__main__':
    # Test the dataset
    train_loader, val_loader, test_loader, vocab_size = get_wikitext_loaders()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, {y.shape}")
