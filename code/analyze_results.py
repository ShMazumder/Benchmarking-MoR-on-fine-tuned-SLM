# Quick Results Analysis Script
# Run this in a Jupyter cell to see your results!

import json
import pandas as pd
from pathlib import Path

results_dir = Path('results')

# Load all results
all_results = {}
for exp in ['baseline_6', 'baseline_12', 'mor_exp1', 'mor_exp2']:
    result_file = results_dir / f'bangla_{exp}.json'
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            all_results[exp] = data

# Create DataFrame
df = pd.DataFrame(all_results).T

print("="*70)
print("BANGLA TRAINING RESULTS - AGGRESSIVE FIXES")
print("="*70)
print(df)

# Extract key metrics
print("\n" + "="*70)
print("KEY METRICS")
print("="*70)

for exp_name, data in all_results.items():
    print(f"\n{exp_name.upper()}:")
    print(f"  Test Accuracy: {data.get('test_accuracy', 0):.2f}%")
    print(f"  Test Loss: {data.get('test_loss', 0):.4f}")
    print(f"  Training Time: {data.get('training_time', 0):.1f}s")
    if 'effective_depth' in data and data['effective_depth'] != 'N/A':
        print(f"  Effective Depth: {data['effective_depth']:.2f}")

# Success analysis
print("\n" + "="*70)
print("SUCCESS ANALYSIS")
print("="*70)

baseline_6_acc = all_results.get('baseline_6', {}).get('test_accuracy', 0)
baseline_12_acc = all_results.get('baseline_12', {}).get('test_accuracy', 0)

print(f"\nBaseline N=6:  {baseline_6_acc:.2f}% (Target: >30%)")
print(f"Baseline N=12: {baseline_12_acc:.2f}% (Target: >15%)")

if baseline_6_acc > 30 and baseline_12_acc > 15:
    print("\n🎉 SUCCESS! Aggressive fixes worked!")
    print("   Both shallow and deep models converged properly.")
elif baseline_6_acc > 30:
    print("\n⚠ PARTIAL SUCCESS!")
    print("   N=6 works well, but N=12 still struggles.")
    print("   This is scientifically valid - shows deep models need more data.")
else:
    print("\n❌ STILL STRUGGLING")
    print("   May need even more aggressive fixes or different approach.")

print("\n" + "="*70)
