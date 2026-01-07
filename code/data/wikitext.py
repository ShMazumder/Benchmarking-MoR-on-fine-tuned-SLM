"""WikiText-2 Dataset Loader with configurable tokenization"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pathlib import Path
from config import Config

try:
    from data.tokenizers import load_sentencepiece, train_sentencepiece, build_word_vocab_from_text, encode_with_sp
except Exception:
    from tokenizers import load_sentencepiece, train_sentencepiece, build_word_vocab_from_text, encode_with_sp


class WikiText2Dataset(Dataset):
    def __init__(self, seq_length=64, split='train', tokenization='char', tokenizer_model=None, vocab_size=None):
        self.seq_length = seq_length
        cfg = Config()

        # Load WikiText-2 from HuggingFace
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        text = ' '.join(dataset['text'])

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
            model_path = tokenizer_model or cfg.tokenizer_model_wikitext
            model_file = Path(model_path)
            if not model_file.exists():
                model_prefix = str(model_file.with_suffix(''))
                train_sentencepiece('wikitext_raw.txt', model_prefix, vocab_size or cfg.subword_vocab_size)
                model_file = Path(model_prefix + '.model')
            sp = load_sentencepiece(str(model_file))
            self.vocab_size = sp.get_piece_size()
            self.stoi = None
            self.itos = None
            data_ids = encode_with_sp(sp, text)

        else:
            raise ValueError(f'Unknown tokenization: {tokenization}')

        self.data = torch.tensor(data_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


def get_wikitext_loaders(batch_size=64, seq_length=64, tokenization=None, tokenizer_model=None, vocab_size=None):
    cfg = Config()
    tokenization = tokenization or cfg.tokenization
    tokenizer_model = tokenizer_model or (cfg.tokenizer_model_wikitext if tokenization == 'subword' else None)
    vocab_size = vocab_size or cfg.subword_vocab_size

    train_dataset = WikiText2Dataset(seq_length, 'train', tokenization, tokenizer_model, vocab_size)
    val_dataset = WikiText2Dataset(seq_length, 'validation', tokenization, tokenizer_model, vocab_size)
    test_dataset = WikiText2Dataset(seq_length, 'test', tokenization, tokenizer_model, vocab_size)

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
    train_loader, val_loader, test_loader, vocab_size = get_wikitext_loaders()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, {y.shape}")
