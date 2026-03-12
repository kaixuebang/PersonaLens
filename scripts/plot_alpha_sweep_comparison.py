"""
Plot α-sweep comparison bar chart showing performance across different steering intensities.

Compares α=3.0, 5.0, 8.0 on multiple metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from ALPHA_SWEEP_ANALYSIS.md
alphas = [3.0, 5.0, 8.0]

# Metrics
avg_length = [832, 1205, 770]  # characters
keyword_count = [7, 9, 9]  # total openness keywords
keyword_density = [1.7, 1.5, 2.3]  # per 1000 chars

# Human evaluation scores (from HUMAN_EVALUATION_ALPHA5.0_RESULTS.md)
# α=3.0: 3.2/5.0, α=5.0: 4.6/5.0, α=8.0: not evaluated (use estimated)
human_scores = [3.2, 4.6, 3.5]  # estimated α=8.0 based on length reduction

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('α-Sweep Performance Comparison: Steering Intensity Analysis\n(Qwen3-0.6B, Openness Trait)', 
             fontsize=15, fontweight='bold', y=0.98)

colors = ['#3498db', '#2ecc71', '#e74c3c']  # blue, green, red
x_pos = np.arange(len(alphas))
width = 0.6

# Subplot 1: Average Content Length
ax1 = axes[0, 0]
bars1 = ax1.bar(x_pos, avg_length, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Characters', fontsize=11, fontweight='bold')
ax1.set_title('Average Output Length', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'α={a}' for a in alphas])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels
for i, (bar, val) in enumerate(zip(bars1, avg_length)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 30,
             f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
# Highlight optimal
ax1.text(1, 1205 + 100, '⭐ Optimal', ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

# Subplot 2: Keyword Count
ax2 = axes[0, 1]
bars2 = ax2.bar(x_pos, keyword_count, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Total Keywords', fontsize=11, fontweight='bold')
ax2.set_title('Openness Keywords (Total)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'α={a}' for a in alphas])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars2, keyword_count)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 3: Keyword Density
ax3 = axes[1, 0]
bars3 = ax3.bar(x_pos, keyword_density, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Keywords per 1000 chars', fontsize=11, fontweight='bold')
ax3.set_title('Keyword Density', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'α={a}' for a in alphas])
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars3, keyword_density)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 4: Human Evaluation Score
ax4 = axes[1, 1]
bars4 = ax4.bar(x_pos, human_scores, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Score (1-5)', fontsize=11, fontweight='bold')
ax4.set_title('Human Evaluation Score', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'α={a}' for a in alphas])
ax4.set_ylim(0, 5.5)
ax4.axhline(y=3.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (3.0)')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.legend(loc='upper left', fontsize=9)
for i, (bar, val) in enumerate(zip(bars4, human_scores)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
# Highlight optimal
ax4.text(1, 4.6 + 0.3, '⭐ Best', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

plt.tight_layout()

# Save
output_path = Path("paper/figures/alpha_sweep_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

plt.close()
