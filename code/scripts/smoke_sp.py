import os
import sys

# Ensure we import tokenizers module directly from code/data
# Run with PYTHONPATH=code/data
import tokenizers

sample_path = os.path.join('code', 'data', 'sample_sp.txt')
dirpath = os.path.dirname(sample_path)
os.makedirs(dirpath, exist_ok=True)
with open(sample_path, 'w', encoding='utf-8') as fh:
    fh.write('hello world\nthis is a tiny sample text for sentencepiece test.\nhello again world\n')

model_prefix = os.path.join('code', 'data', 'tokenizers', 'smoke_test_sp')
print('Training SentencePiece model (small sample) ->', model_prefix + '.model')
try:
    tokenizers.train_sentencepiece(sample_path, model_prefix, vocab_size=100)
except Exception as e:
    print('SentencePiece training failed:', e)
    raise

sp = tokenizers.load_sentencepiece(model_prefix + '.model')
print('Trained vocab size:', sp.get_piece_size())
print('Example encoding for "hello world":', tokenizers.encode_with_sp(sp, 'hello world'))
