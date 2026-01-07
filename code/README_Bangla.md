# Running Bangla SLM experiments (MoR benchmark)

This document explains how to prepare data and run the MoR benchmarking scripts for the Bangla SLM dataset on Kaggle or local machines.

1) Prepare the dataset

- Place your Bangla SLM text file at `data/bangla/bangla_slm.txt` (UTF-8, plain text). The loader expects one long text file.
- Alternatively, add the dataset to the Kaggle dataset for the notebook and copy it into `data/bangla/` during notebook setup.

Example (Kaggle notebook cell to create directory and copy dataset):

```bash
mkdir -p /kaggle/working/data/bangla
# If you uploaded a dataset to Kaggle as `USERNAME/bangla-slm`, adapt this path accordingly.
# Example: copy from dataset folder to working dir
# cp -r /kaggle/input/bangla-slm/* /kaggle/working/data/bangla/
```

2) Tokenization options

- `char` : character-level (default)
- `word` : simple whitespace word-level vocabulary
- `subword` : SentencePiece BPE subword tokens (recommended for SLM tasks)

If using `subword` the code will train a SentencePiece model from `data/bangla/bangla_slm.txt` and store it at `data/tokenizers/sp_bangla.model` by default.

3) Example commands

Single-process (Kaggle single GPU / CPU):

```bash
# CPU or single GPU (Kaggle GPU recommended)
PYTHONPATH=code python code/train_amp.py \
  --dataset bangla \
  --experiment mor_exp1 \
  --tokenization subword \
  --subword_vocab_size 8000 \
  --amp \
  --device cuda \
  --epochs 3
```

Multi-GPU (local cluster with torchrun):

```bash
# Replace NUM with number of GPUs per node
torchrun --nproc_per_node=NUM python code/train_amp.py \
  --dataset bangla \
  --experiment mor_exp1 \
  --tokenization subword \
  --subword_vocab_size 8000 \
  --amp \
  --syncbn \
  --epochs 3
```

Notes:
- On Kaggle, `torchrun` is not available for multi-GPU (Kaggle provides single GPU). Use the `python` invocation.
- If you want deterministic splitting or reproducible runs, set seeds in `config.py` or pass them via a wrapper.

4) Outputs

- Checkpoints are written to `checkpoints/` (per-epoch files) and results are saved to `results/`.
- Per-epoch history JSON files are stored as `{experiment}_history.json` inside `results/`.

5) Troubleshooting

- Ensure `sentencepiece` is installed (added to `code/requirements.txt`).
- If SentencePiece training is slow on the full dataset, run a smaller `vocab_size` for quick tests and increase for final runs.

If you want, I can add a Kaggle notebook cell that automates copying the dataset from the Kaggle dataset mount, creating the tokenizer, and running a short test (1–3 epochs). Would you like that?