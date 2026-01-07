"""Bangla SLM Dataset Loader with configurable tokenization"""
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

try:
    from data.tokenizers import load_sentencepiece, train_sentencepiece, build_word_vocab_from_text, encode_with_sp
except Exception:
    from tokenizers import load_sentencepiece, train_sentencepiece, build_word_vocab_from_text, encode_with_sp


class BanglaSLMDataset(Dataset):
    def __init__(self, seq_length=64, split='train', split_ratio=0.9, tokenization='char', tokenizer_model=None, vocab_size=None, data_file=None):
        self.seq_length = seq_length
        cfg = Config()

        # Determine data path: use argument if provided, else use config default
        if data_file:
            data_path = Path(data_file)
        else:
            # Fallback to config 
            data_path = Path(cfg.bangla_data_file)
            
            # Ensure the directory exists if we are likely to write to it (though we just read usually)
            if not data_path.parent.exists():
                 data_path.parent.mkdir(parents=True, exist_ok=True)

        if not data_path.exists():
            raise FileNotFoundError(f"Bangla dataset file not found: {data_path}. Please place a UTF-8 text file at that path.")

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
            model_path = tokenizer_model or cfg.tokenizer_model_bangla
            model_file = Path(model_path)
            
            # Ensure tokenizer directory exists
            model_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not model_file.exists():
                print(f"Training SentencePiece model for Bangla at {model_file}...")
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

        data = torch.tensor(data_ids, dtype=torch.long)

        split_idx = int(len(data) * split_ratio)
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

    def __len__(self):
        # We need at least seq_length + 1 tokens to form one sample (x, y)
        if len(self.data) <= self.seq_length:
            return 0
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


def get_bangla_loaders(batch_size=64, seq_length=64, split_ratio=0.9, tokenization=None, tokenizer_model=None, vocab_size=None, data_file=None):
    cfg = Config()
    tokenization = tokenization or cfg.tokenization
    tokenizer_model = tokenizer_model or (cfg.tokenizer_model_bangla if tokenization == 'subword' else None)
    vocab_size = vocab_size or cfg.subword_vocab_size

    # Pass data_file to the dataset class
    train_dataset = BanglaSLMDataset(seq_length=seq_length, split='train', split_ratio=split_ratio,
                                     tokenization=tokenization, tokenizer_model=tokenizer_model, vocab_size=vocab_size, data_file=data_file)
    test_dataset = BanglaSLMDataset(seq_length=seq_length, split='test', split_ratio=split_ratio,
                                    tokenization=tokenization, tokenizer_model=tokenizer_model, vocab_size=vocab_size, data_file=data_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_dataset.vocab_size


if __name__ == '__main__':
    # quick self-check
    try:
        cfg = Config()
        print(f"Checking for Bangla data at: {cfg.bangla_data_file}")
        if os.path.exists(cfg.bangla_data_file):
            tr, te, vs = get_bangla_loaders(batch_size=8, seq_length=64)
            print('Bangla vocab size:', vs)
            if len(tr) > 0:
                x, y = next(iter(tr))
                print('Batch shapes:', x.shape, y.shape)
            else:
                print("Dataset too small to generate batches.")
        else:
            print("Bangla data file not found. Skipping load test.")
    except Exception as e:
        print('Bangla loader test failed:', e)