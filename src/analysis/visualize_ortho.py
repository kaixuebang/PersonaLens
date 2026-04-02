import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load matrices
model = "Qwen_Qwen2.5-0.5B-Instruct"
analysis_dir = Path("results/analysis")

frameworks = {
    "Big Five": (
        "ortho_matrix_big_five_",
        [
            "Openness",
            "Conscientiousness",
            "Extraversion",
            "Agreeableness",
            "Neuroticism",
        ],
    ),
    "MBTI": ("ortho_matrix_mbti_", ["Extraversion", "Sensing", "Thinking", "Judging"]),
    "Jungian": (
        "ortho_matrix_jungian_",
        ["Ni", "Ne", "Si", "Se", "Ti", "Te", "Fi", "Fe"],
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (fw_name, (prefix, labels)) in enumerate(frameworks.items()):
    matrix_file = analysis_dir / f"{prefix}{model}.npy"
    if matrix_file.exists():
        matrix = np.load(matrix_file)

        ax = axes[idx]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "|cos θ|"},
        )
        ax.set_title(
            f"{fw_name}\n({len(labels)} dimensions)", fontsize=12, fontweight="bold"
        )

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

plt.suptitle(
    f"Cross-Framework Orthogonality Comparison\n{model}",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    "results/analysis/orthogonality_comparison.png", dpi=150, bbox_inches="tight"
)
print("Saved: results/analysis/orthogonality_comparison.png")

# Summary statistics visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

framework_names = ["Big Five", "MBTI", "Jungian"]
mean_ortho = [0.4179, 0.2006, 0.2374]
std_ortho = [0.1312, 0.1760, 0.1793]

# Bar plot
bars = ax1.bar(
    framework_names,
    mean_ortho,
    yerr=std_ortho,
    color=["#e74c3c", "#27ae60", "#3498db"],
    alpha=0.7,
    capsize=5,
    edgecolor="black",
    linewidth=1.5,
)
ax1.set_ylabel("Mean |cos θ| (lower = more orthogonal)", fontsize=11)
ax1.set_title("Framework Orthogonality Comparison", fontsize=12, fontweight="bold")
ax1.axhline(
    y=0.2, color="green", linestyle="--", alpha=0.5, label="Near-orthogonal threshold"
)
ax1.legend()
ax1.set_ylim(0, 0.5)

# Add value labels
for bar, mean, std in zip(bars, mean_ortho, std_ortho):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + std + 0.01,
        f"{mean:.3f}±{std:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Number of dimensions vs orthogonality
ax2.scatter(
    [5, 4, 8],
    mean_ortho,
    s=200,
    c=["#e74c3c", "#27ae60", "#3498db"],
    alpha=0.7,
    edgecolors="black",
    linewidth=2,
)
for i, fw in enumerate(framework_names):
    ax2.annotate(
        fw,
        (5 - i, mean_ortho[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )
ax2.set_xlabel("Number of Dimensions", fontsize=11)
ax2.set_ylabel("Mean |cos θ|", fontsize=11)
ax2.set_title("Dimensionality vs Orthogonality", fontsize=12, fontweight="bold")
ax2.set_xlim(3, 9)
ax2.set_ylim(0.15, 0.45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/analysis/orthogonality_summary.png", dpi=150, bbox_inches="tight")
print("Saved: results/analysis/orthogonality_summary.png")

print("\nKey Findings:")
print("1. MBTI (4 binary dimensions): Cleanest geometric structure")
print("2. Jungian (8 functions): Good separation despite higher dimensionality")
print("3. Big Five (5 continuous factors): Most overlap between dimensions")
print("\nInterpretation: Binary typologies (MBTI/Jungian) may produce more")
print("orthogonal representations than continuous factor models (Big Five)")
