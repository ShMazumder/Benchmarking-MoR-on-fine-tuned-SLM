# MoR Benchmarking Implementation

Complete PyTorch implementation of the "Benchmarking Mixture-of-Recursion (MoR) for Character-Level Language Modeling" paper.

## Project Structure

```
code/
├── data/
│   ├── shakespeare.py      # Tiny Shakespeare dataset loader
│   └── wikitext.py         # WikiText-2 dataset loader
├── models/
│   ├── transformer.py      # Baseline Transformer
│   ├── mor_transformer.py  # MoR Transformer
│   └── router.py           # MoR Router module
├── train.py                # Training script
├── config.py               # Configuration
├── utils.py                # Utility functions
└── requirements.txt        # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Experiment 1: Efficiency Profiling (N=12 vs MoR N=12)

**Baseline (N=12):**
```bash
python train.py --dataset shakespeare --experiment baseline_12
```

**MoR (N=12):**
```bash
python train.py --dataset shakespeare --experiment mor_exp1
```

### Experiment 2: Equal Cost (N=6 vs MoR N=12, E≈6)

**Baseline (N=6):**
```bash
python train.py --dataset shakespeare --experiment baseline_6
```

**MoR (N=12, E≈6):**
```bash
python train.py --dataset shakespeare --experiment mor_exp2
```

### WikiText-2 Experiments

Replace `--dataset shakespeare` with `--dataset wikitext`:

```bash
python train.py --dataset wikitext --experiment baseline_12
python train.py --dataset wikitext --experiment mor_exp1
python train.py --dataset wikitext --experiment baseline_6
python train.py --dataset wikitext --experiment mor_exp2
```

## Configuration

Edit `config.py` to modify hyperparameters:

- `d_model`: Embedding dimension (default: 256)
- `n_heads`: Number of attention heads (default: 8)
- `d_ff`: Feedforward dimension (default: 2048)
- `batch_size`: Batch size (default: 64)
- `learning_rate`: Learning rate (default: 1e-3)
- `lambda_depth_penalty`: Depth penalty weight (default: 0.1)

## Results

Results are saved to `results/` directory as JSON files:
- `shakespeare_baseline_12.json`
- `shakespeare_mor_exp1.json`
- `wikitext_baseline_12.json`
- etc.

## Model Architecture

### Baseline Transformer
- Fixed-depth Transformer with N layers
- All tokens pass through all layers
- Effective depth = N

### MoR Transformer
- Shared recursive layer with routing
- Router decides: Skip (cost 0), Forward (cost 1), or Recurse (cost 1+)
- Effective depth varies per token based on routing decisions

### Router Architecture
- Input: Token embeddings (256-dim)
- MLP: 256 → 128 → 3 (with ReLU)
- Output: Softmax over {Skip, Forward, Recurse}
- Training: Gumbel-Softmax (τ=1.0)
- Inference: Argmax selection

## Expected Results

### Tiny Shakespeare

**Experiment 1:**
- Baseline N=12: Acc ≈ 22.76%, E=12.00, Time ≈ 108s
- MoR N=12: Acc ≈ 22.76%, E=8.00, Time ≈ 83s

**Experiment 2:**
- Baseline N=6: Test Acc ≈ 39.87%, E=6.00, Time ≈ 30s
- MoR N=12: Test Acc ≈ 49.67%, E=5.89, Time ≈ 59s

## Hardware Requirements

- GPU: NVIDIA GPU with 4GB+ VRAM (recommended)
- CPU: Works but slower
- RAM: 8GB+

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{your2026benchmarking,
  title={Benchmarking Mixture-of-Recursion (MoR) for Character-Level Language Modeling},
  author={Your Name},
  booktitle={IEEE QPAIN},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

- MoR architecture: Bae et al. (2024)
- Tiny Shakespeare dataset: Andrej Karpathy
- WikiText-2 dataset: Merity et al. (2016)
