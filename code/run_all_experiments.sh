#!/bin/bash
# Run all experiments for both datasets (AMP + optional DDP)

set -e

echo "========================================="
echo "Running MoR Benchmarking Experiments (AMP + optional DDP)"
echo "========================================="

# Create results directory
mkdir -p results

# Determine GPU count (can override by setting NUM_GPUS env)
NUM_GPUS=${NUM_GPUS:-0}
if command -v nvidia-smi >/dev/null 2>&1; then
	GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || true)
	if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 0 ]; then
		NUM_GPUS=$GPU_COUNT
	fi
fi

echo "Using NUM_GPUS=$NUM_GPUS"

run_script() {
	# Args: script and its args (no leading 'python')
	if [ "$NUM_GPUS" -gt 1 ]; then
		echo "Launching with torchrun (nproc_per_node=$NUM_GPUS): python $*"
		torchrun --nproc_per_node=$NUM_GPUS python $@
	else
		echo "Running single-process: python $*"
		python $@
	fi
}

# Tiny Shakespeare Experiments
echo "\n=== Tiny Shakespeare Experiments ==="

echo "\n[1/4] Training Baseline N=12..."
run_script train_amp.py --dataset shakespeare --experiment baseline_12 --amp

echo "\n[2/4] Training MoR Exp1 (N=12)..."
run_script train_amp.py --dataset shakespeare --experiment mor_exp1 --amp

echo "\n[3/4] Training Baseline N=6..."
run_script train_amp.py --dataset shakespeare --experiment baseline_6 --amp

echo "\n[4/4] Training MoR Exp2 (N=12, E≈6)..."
run_script train_amp.py --dataset shakespeare --experiment mor_exp2 --amp

# WikiText-2 Experiments
echo "\n=== WikiText-2 Experiments ==="

echo "\n[1/4] Training Baseline N=12..."
run_script train_amp.py --dataset wikitext --experiment baseline_12 --amp

echo "\n[2/4] Training MoR Exp1 (N=12)..."
run_script train_amp.py --dataset wikitext --experiment mor_exp1 --amp

echo "\n[3/4] Training Baseline N=6..."
run_script train_amp.py --dataset wikitext --experiment baseline_6 --amp

echo "\n[4/4] Training MoR Exp2 (N=12, E≈6)..."
run_script train_amp.py --dataset wikitext --experiment mor_exp2 --amp

echo "\n========================================="
echo "All experiments completed!"
echo "Results saved to results/ directory"
echo "========================================="
