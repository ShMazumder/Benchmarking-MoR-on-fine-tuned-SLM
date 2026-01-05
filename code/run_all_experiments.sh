#!/bin/bash
# Run all experiments for both datasets

echo "========================================="
echo "Running MoR Benchmarking Experiments"
echo "========================================="

# Create results directory
mkdir -p results

# Tiny Shakespeare Experiments
echo "\n=== Tiny Shakespeare Experiments ==="

echo "\n[1/4] Training Baseline N=12..."
python train.py --dataset shakespeare --experiment baseline_12

echo "\n[2/4] Training MoR Exp1 (N=12)..."
python train.py --dataset shakespeare --experiment mor_exp1

echo "\n[3/4] Training Baseline N=6..."
python train.py --dataset shakespeare --experiment baseline_6

echo "\n[4/4] Training MoR Exp2 (N=12, E≈6)..."
python train.py --dataset shakespeare --experiment mor_exp2

# WikiText-2 Experiments
echo "\n=== WikiText-2 Experiments ==="

echo "\n[1/4] Training Baseline N=12..."
python train.py --dataset wikitext --experiment baseline_12

echo "\n[2/4] Training MoR Exp1 (N=12)..."
python train.py --dataset wikitext --experiment mor_exp1

echo "\n[3/4] Training Baseline N=6..."
python train.py --dataset wikitext --experiment baseline_6

echo "\n[4/4] Training MoR Exp2 (N=12, E≈6)..."
python train.py --dataset wikitext --experiment mor_exp2

echo "\n========================================="
echo "All experiments completed!"
echo "Results saved to results/ directory"
echo "========================================="
