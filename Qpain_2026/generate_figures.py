import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create figures directory
script_dir = Path(__file__).parent
figures_dir = script_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# Figure 1: Experiment 1 - Efficiency Comparison (N=12)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy
models = ['Standard\n(N=12)', 'MoR\n(N=12)']
accuracy = [22.76, 22.76]
colors = ['#FF6B6B', '#4ECDC4']
axes[0].bar(models, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('(a) Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 30])
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(accuracy):
    axes[0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

# Effective Depth
depth = [12.00, 8.00]
axes[1].bar(models, depth, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Effective Depth (Layers)', fontsize=12, fontweight='bold')
axes[1].set_title('(b) Computational Depth', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 14])
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(depth):
    axes[1].text(i, v + 0.3, f'{v:.2f}', ha='center', fontweight='bold')
# Add savings annotation
axes[1].annotate('33% Reduction', xy=(1, 8), xytext=(0.5, 10),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=11, fontweight='bold', color='green')

# Training Time
time = [108, 83]
axes[2].bar(models, time, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[2].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
axes[2].set_title('(c) Training Efficiency', fontsize=13, fontweight='bold')
axes[2].set_ylim([0, 130])
axes[2].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(time):
    axes[2].text(i, v + 2, f'{v}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'exp1_efficiency_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: Experiment 2 - Equal Cost Comparison
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models2 = ['Baseline\n(N=6)', 'MoR\n(N=12, E≈6)']
colors2 = ['#FF6B6B', '#4ECDC4']

# Held-out Accuracy
accuracy2 = [14.90, 49.67]
axes[0].bar(models2, accuracy2, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Held-out Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('(a) Generalization Performance', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 60])
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(accuracy2):
    axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
# Add improvement annotation
axes[0].annotate('+34.77%\nImprovement', xy=(1, 49.67), xytext=(0.5, 55),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=11, fontweight='bold', color='green')

# Effective Depth
depth2 = [6.00, 5.89]
axes[1].bar(models2, depth2, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Effective Depth (Layers)', fontsize=12, fontweight='bold')
axes[1].set_title('(b) Computational Cost', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 8])
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(depth2):
    axes[1].text(i, v + 0.15, f'{v:.2f}', ha='center', fontweight='bold')

# Training Time
time2 = [30, 59]
axes[2].bar(models2, time2, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[2].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
axes[2].set_title('(c) Training Time', fontsize=13, fontweight='bold')
axes[2].set_ylim([0, 70])
axes[2].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(time2):
    axes[2].text(i, v + 1.5, f'{v}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'exp2_equal_cost_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Multi-Granularity Performance Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

datasets = ['Shakespeare\n(Char)', 'Bangla\n(Subword)', 'WikiText-2\n(Subword)']
x = np.arange(len(datasets))
width = 0.25

# Data for each experiment type
baseline_n12 = [22.76, 75.46, 3.69]
mor_n12 = [22.76, 48.29, 30.06]
baseline_n6 = [15.27, 63.74, 31.89]

bars1 = ax.bar(x - width, baseline_n12, width, label='Baseline (N=12)', 
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, mor_n12, width, label='MoR (N=12)', 
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, baseline_n6, width, label='Baseline (N=6)', 
               color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Dataset (Granularity)', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance Across Token Granularities', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 85])

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'multi_granularity_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: Effective Depth vs Accuracy Trade-off
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Data points: (Effective Depth, Accuracy, Model Name)
data_points = [
    (12.00, 22.76, 'Baseline N=12\n(Shakespeare)', '#FF6B6B', 'o'),
    (8.00, 22.76, 'MoR N=12\n(Shakespeare)', '#4ECDC4', 's'),
    (6.00, 14.90, 'Baseline N=6\n(Shakespeare)', '#FF6B6B', '^'),
    (5.89, 49.67, 'MoR N=12 E≈6\n(Shakespeare)', '#4ECDC4', 'D'),
    (12.00, 3.69, 'Baseline N=12\n(WikiText)', '#FF6B6B', 'o'),
    (8.13, 30.06, 'MoR N=12\n(WikiText)', '#4ECDC4', 's'),
    (12.00, 75.46, 'Baseline N=12\n(Bangla)', '#FF6B6B', 'v'),
    (8.28, 48.29, 'MoR N=12\n(Bangla)', '#4ECDC4', 'X'),
]

for depth, acc, label, color, marker in data_points:
    ax.scatter(depth, acc, s=200, c=color, marker=marker, 
              edgecolors='black', linewidths=2, alpha=0.8, label=label)

ax.set_xlabel('Effective Depth (Layers)', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Efficiency-Performance Trade-off', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([4, 13])
ax.set_ylim([0, 85])

plt.tight_layout()
plt.savefig(figures_dir / 'depth_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 5: Stability Analysis (WikiText-2)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

models_stability = ['Baseline\nN=6', 'Baseline\nN=12', 'MoR\nN=12']
accuracy_stability = [31.89, 3.69, 30.06]
colors_stability = ['#95E1D3', '#FF6B6B', '#4ECDC4']
patterns = ['', '///', '']

bars = ax.bar(models_stability, accuracy_stability, color=colors_stability, 
              alpha=0.8, edgecolor='black', linewidth=2)

# Add hatching to failed model
bars[1].set_hatch('///')

ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Training Stability on WikiText-2 (High-Density Subword)', 
             fontsize=14, fontweight='bold')
ax.set_ylim([0, 40])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, v) in enumerate(zip(bars, accuracy_stability)):
    label = f'{v:.2f}%'
    if i == 1:
        label += '\n(Diverged)'
        ax.text(i, v + 1, label, ha='center', fontweight='bold', 
               color='red', fontsize=11)
    else:
        ax.text(i, v + 1, label, ha='center', fontweight='bold', fontsize=11)

# Add annotation for stability recovery
ax.annotate('MoR Recovers\nStability', xy=(2, 30), xytext=(1.5, 35),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'),
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(figures_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 6: Computational Savings Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

experiments = ['Exp 1\n(Shakespeare)', 'Exp 3\n(Bangla)', 'Exp 4\n(WikiText)']
depth_reduction = [33.3, 32.1, 32.3]  # Percentage reduction
colors_savings = ['#4ECDC4', '#4ECDC4', '#4ECDC4']

bars = ax.bar(experiments, depth_reduction, color=colors_savings, 
              alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('Depth Reduction (%)', fontsize=13, fontweight='bold')
ax.set_title('MoR Computational Savings Across Datasets', 
             fontsize=14, fontweight='bold')
ax.set_ylim([0, 40])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=33, color='red', linestyle='--', linewidth=2, alpha=0.5, label='~33% Average')

# Add value labels
for bar, v in zip(bars, depth_reduction):
    ax.text(bar.get_x() + bar.get_width()/2., v + 0.8,
            f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)

ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(figures_dir / 'computational_savings.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ All figures generated successfully!")
print(f"✓ Figures saved to: {figures_dir.absolute()}")
print("\nGenerated figures:")
print("  1. exp1_efficiency_comparison.png")
print("  2. exp2_equal_cost_comparison.png")
print("  3. multi_granularity_summary.png")
print("  4. depth_accuracy_tradeoff.png")
print("  5. stability_analysis.png")
print("  6. computational_savings.png")
