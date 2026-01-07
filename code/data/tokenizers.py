import os
from pathlib import Path
import sentencepiece as spm


def ensure_tokenizer_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def train_sentencepiece(input_text_path: str, model_prefix: str, vocab_size: int):
    ensure_tokenizer_dir(Path(model_prefix).parent)
    cmd = f"--input={input_text_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0"
    spm.SentencePieceTrainer.Train(cmd)
    return model_prefix + '.model'


def load_sentencepiece(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def encode_with_sp(sp, text: str):
    return sp.EncodeAsIds(text)


def build_word_vocab_from_text(text: str):
    # simple whitespace-tokenizer fallback
    words = text.split()
    uniq = []
    seen = set()
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    stoi = {w: i for i, w in enumerate(uniq)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos
