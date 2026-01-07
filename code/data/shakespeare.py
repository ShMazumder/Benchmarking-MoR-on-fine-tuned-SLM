"""Tiny Shakespeare Dataset Loader with configurable tokenization"""
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os
from pathlib import Path
from config import Config

try:
    from data.tokenizers import load_sentencepiece, train_sentencepiece, build_word_vocab_from_text, encode_with_sp
except Exception:
    from tokenizers import load_sentencepiece, train_sentencepiece, build_word_vocab_from_text, encode_with_sp


class TinyShakespeareDataset(Dataset):
    def __init__(self, seq_length=64, split='train', split_ratio=0.9, tokenization='char', tokenizer_model=None, vocab_size=None):
        self.seq_length = seq_length
        cfg = Config()

        # Download or load data
        data_dir = Path(cfg.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / 'shakespeare.txt'
        if not data_path.exists():
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            import requests
            response = requests.get(url)
            data_path.write_text(response.text, encoding='utf-8')

        text = data_path.read_text(encoding='utf-8')

        self.tokenization = tokenization
        if tokenization == 'char':
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            data_ids = [self.stoi[ch] for ch in text]

        elif tokenization == 'word':
            stoi, itos = build_word_vocab_from_text(text)
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
            data_ids = [self.stoi[w] for w in text.split()]

        elif tokenization == 'subword':
            # train or load sentencepiece model
            model_path = tokenizer_model or cfg.tokenizer_model_shakespeare
            model_file = Path(model_path)
            if not model_file.exists():
                model_prefix = str(model_file.with_suffix(''))
                train_sentencepiece(str(data_path), model_prefix, vocab_size or cfg.subword_vocab_size)
                model_file = Path(model_prefix + '.model')
            sp = load_sentencepiece(str(model_file))
            self.vocab_size = sp.get_piece_size()
            self.stoi = None
            self.itos = None
            data_ids = encode_with_sp(sp, text)

        else:
            raise ValueError(f'Unknown tokenization: {tokenization}')

        import torch
        data = torch.tensor(data_ids, dtype=torch.long)

        # Split data
        split_idx = int(len(data) * split_ratio)
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


def get_shakespeare_loaders(batch_size=64, seq_length=64, split_ratio=0.9, tokenization=None, tokenizer_model=None, vocab_size=None):
    cfg = Config()
    tokenization = tokenization or cfg.tokenization
    tokenizer_model = tokenizer_model or (cfg.tokenizer_model_shakespeare if tokenization == 'subword' else None)
    vocab_size = vocab_size or cfg.subword_vocab_size

    train_dataset = TinyShakespeareDataset(seq_length, 'train', split_ratio, tokenization, tokenizer_model, vocab_size)
    test_dataset = TinyShakespeareDataset(seq_length, 'test', split_ratio, tokenization, tokenizer_model, vocab_size)

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
    train_loader, test_loader, vocab_size = get_shakespeare_loaders()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, {y.shape}")
