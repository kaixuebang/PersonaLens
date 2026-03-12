"""
Plot BFI-44 dose-response curves for all 5 Big Five traits.

Shows how BFI scores change with steering intensity (α).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load BFI results for all 5 traits
traits = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
model_name = "Qwen_Qwen3-0.6B"
bfi_dir = Path("bfi_results") / model_name

# Collect data
data = {}
for trait in traits:
    filepath = bfi_dir / f"bfi_self_report_{trait}.json"
    if filepath.exists():
        with open(filepath) as f:
            result = json.load(f)
            alphas = result["alphas"]
            scores = [result["results"][str(alpha)]["trait_score"] for alpha in alphas]
            data[trait] = (alphas, scores)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each trait
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
markers = ["o", "s", "^", "D", "v"]

for trait, color, marker in zip(traits, colors, markers):
    if trait in data:
        alphas, scores = data[trait]
        ax.plot(
            alphas,
            scores,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8,
            label=trait.capitalize(),
            alpha=0.8,
        )

# Styling
ax.axhline(
    y=3.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Neutral (3.0)"
)
ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Steering Intensity (α)", fontsize=13, fontweight="bold")
ax.set_ylabel("BFI-44 Trait Score (1-5)", fontsize=13, fontweight="bold")
ax.set_title(
    "BFI-44 Dose-Response Curves: Steering Intensity vs Personality Scores\n(Qwen3-0.6B)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.legend(loc="best", frameon=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
ax.set_xlim(-9, 9)
ax.set_ylim(2.4, 3.2)

# Add annotations for key findings
ax.text(
    6,
    3.05,
    "Extraversion & Neuroticism:\nStrong dose-response",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    ha="center",
)
ax.text(
    -6,
    2.92,
    "Agreeableness:\nMinimal steering effect",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    ha="center",
)

plt.tight_layout()

# Save
output_path = Path("paper/figures/bfi_dose_response.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_path}")

plt.close()
