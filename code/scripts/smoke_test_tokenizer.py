from config import Config
from data.shakespeare import get_shakespeare_loaders

if __name__ == '__main__':
    cfg = Config()
    print('Config tokenization default:', cfg.tokenization)
    train_loader, test_loader, vocab_size = get_shakespeare_loaders(
        batch_size=8,
        seq_length=64,
        tokenization='subword',
        vocab_size=2000
    )
    print('Vocab size:', vocab_size)
    x, y = next(iter(train_loader))
    print('Batch x shape:', x.shape)
    print('Batch y shape:', y.shape)
