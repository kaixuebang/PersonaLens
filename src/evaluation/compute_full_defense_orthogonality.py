"""
Compute full 9-defense-mechanism orthogonality matrix for Qwen2.5-0.5B-Instruct.
Includes Vaillant hierarchy annotations and creates publication-quality visualization.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Vaillant's hierarchy classification
VAILLANT_HIERARCHY = {
    "humor": "mature",
    "sublimation": "mature",
    "rationalization": "neurotic",
    "intellectualization": "neurotic",
    "displacement": "neurotic",
    "projection": "immature",
    "denial": "immature",
    "regression": "immature",
    "reaction_formation": "immature",
}

# Order mechanisms by hierarchy for visualization
MECHANISM_ORDER = [
    # Mature
    "humor",
    "sublimation",
    # Neurotic
    "rationalization",
    "intellectualization",
    "displacement",
    # Immature
    "projection",
    "denial",
    "regression",
    "reaction_formation",
]

MECHANISM_LABELS = {
    "humor": "Humor",
    "sublimation": "Sublimation",
    "rationalization": "Rationalization",
    "intellectualization": "Intellectualization",
    "displacement": "Displacement",
    "projection": "Projection",
    "denial": "Denial",
    "regression": "Regression",
    "reaction_formation": "Reaction Formation",
}


def load_persona_vector(model_name, mechanism, layer):
    """Load persona vector for a given mechanism at specified layer."""
    base_path = Path(
        f"/home/fqwqf/persona/persona_vectors/{model_name}/{mechanism}/vectors"
    )
    vector_file = base_path / f"mean_diff_layer_{layer}.npy"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector not found: {vector_file}")

    vector = np.load(vector_file)
    # Normalize
    return vector / np.linalg.norm(vector)


def compute_orthogonality_matrix(model_name, layer, mechanisms):
    """Compute cosine similarity matrix for all mechanism pairs."""
    n = len(mechanisms)
    cos_matrix = np.zeros((n, n))

    # Load all vectors
    vectors = {}
    for mech in mechanisms:
        try:
            vectors[mech] = load_persona_vector(model_name, mech, layer)
        except FileNotFoundError:
            print(f"Warning: Vector not found for {mech}, skipping")
            return None

    # Compute pairwise cosines
    for i, mech_i in enumerate(mechanisms):
        for j, mech_j in enumerate(mechanisms):
            if mech_i in vectors and mech_j in vectors:
                cos_matrix[i, j] = np.dot(vectors[mech_i], vectors[mech_j])

    return cos_matrix


def plot_orthogonality_matrix(cos_matrix, mechanisms, save_path):
    """Create publication-quality orthogonality visualization with hierarchy annotations."""
    fig, ax = plt.subplots(figsize=(10, 9))

    # Create labels with hierarchy markers
    labels = []
    for mech in mechanisms:
        hierarchy = VAILLANT_HIERARCHY[mech]
        marker = {"mature": "★", "neurotic": "◆", "immature": "●"}[hierarchy]
        labels.append(f"{marker} {MECHANISM_LABELS[mech]}")

    # Plot heatmap
    im = ax.imshow(np.abs(cos_matrix), cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(mechanisms)))
    ax.set_yticks(np.arange(len(mechanisms)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Cosine Similarity|", rotation=270, labelpad=20, fontsize=11)

    # Add text annotations
    for i in range(len(mechanisms)):
        for j in range(len(mechanisms)):
            if i != j:  # Skip diagonal
                text = ax.text(
                    j,
                    i,
                    f"{np.abs(cos_matrix[i, j]):.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # Add hierarchy dividers
    # After mature (index 2)
    ax.axhline(y=1.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.axvline(x=1.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    # After neurotic (index 5)
    ax.axhline(y=4.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.axvline(x=4.5, color="black", linewidth=2, linestyle="--", alpha=0.5)

    # Add hierarchy labels
    ax.text(
        -1.5, 0.5, "Mature", rotation=90, va="center", fontsize=11, fontweight="bold"
    )
    ax.text(
        -1.5, 3, "Neurotic", rotation=90, va="center", fontsize=11, fontweight="bold"
    )
    ax.text(
        -1.5, 6.5, "Immature", rotation=90, va="center", fontsize=11, fontweight="bold"
    )

    ax.set_title(
        "Defense Mechanism Orthogonality Matrix (Qwen2.5-0.5B, Layer 12)\nOrganized by Vaillant Hierarchy",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="black",
            markersize=12,
            label="Mature",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label="Neurotic",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label="Immature",
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=10
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")
    plt.close()


def compute_statistics(cos_matrix, mechanisms):
    """Compute clustering statistics by hierarchy level."""
    n = len(mechanisms)

    # Get hierarchy indices
    mature_idx = [
        i for i, m in enumerate(mechanisms) if VAILLANT_HIERARCHY[m] == "mature"
    ]
    neurotic_idx = [
        i for i, m in enumerate(mechanisms) if VAILLANT_HIERARCHY[m] == "neurotic"
    ]
    immature_idx = [
        i for i, m in enumerate(mechanisms) if VAILLANT_HIERARCHY[m] == "immature"
    ]

    def get_off_diag_values(indices):
        """Get off-diagonal values within a group."""
        values = []
        for i in indices:
            for j in indices:
                if i < j:  # Upper triangle only
                    values.append(np.abs(cos_matrix[i, j]))
        return values

    def get_between_group_values(idx1, idx2):
        """Get values between two groups."""
        values = []
        for i in idx1:
            for j in idx2:
                values.append(np.abs(cos_matrix[i, j]))
        return values

    stats = {
        "within_mature": {
            "mean": np.mean(get_off_diag_values(mature_idx))
            if len(mature_idx) > 1
            else 0,
            "values": get_off_diag_values(mature_idx),
        },
        "within_neurotic": {
            "mean": np.mean(get_off_diag_values(neurotic_idx))
            if len(neurotic_idx) > 1
            else 0,
            "values": get_off_diag_values(neurotic_idx),
        },
        "within_immature": {
            "mean": np.mean(get_off_diag_values(immature_idx))
            if len(immature_idx) > 1
            else 0,
            "values": get_off_diag_values(immature_idx),
        },
        "between_mature_neurotic": {
            "mean": np.mean(get_between_group_values(mature_idx, neurotic_idx)),
            "values": get_between_group_values(mature_idx, neurotic_idx),
        },
        "between_mature_immature": {
            "mean": np.mean(get_between_group_values(mature_idx, immature_idx)),
            "values": get_between_group_values(mature_idx, immature_idx),
        },
        "between_neurotic_immature": {
            "mean": np.mean(get_between_group_values(neurotic_idx, immature_idx)),
            "values": get_between_group_values(neurotic_idx, immature_idx),
        },
        "overall_mean": np.mean(np.abs(cos_matrix[np.triu_indices(n, k=1)])),
    }

    return stats


def main():
    model_name = "Qwen_Qwen2.5-0.5B-Instruct"
    layer = 12  # Best layer for Qwen2.5
    mechanisms = MECHANISM_ORDER

    print(f"Computing orthogonality matrix for {model_name} at layer {layer}...")
    print(f"Mechanisms: {mechanisms}")

    # Compute matrix
    cos_matrix = compute_orthogonality_matrix(model_name, layer, mechanisms)

    if cos_matrix is None:
        print("Failed to compute matrix - missing vectors")
        return

    # Compute statistics
    stats = compute_statistics(cos_matrix, mechanisms)

    print("\n=== Clustering Statistics by Vaillant Hierarchy ===")
    print(
        f"Within Mature (n={len(stats['within_mature']['values'])} pairs): {stats['within_mature']['mean']:.3f}"
    )
    print(
        f"Within Neurotic (n={len(stats['within_neurotic']['values'])} pairs): {stats['within_neurotic']['mean']:.3f}"
    )
    print(
        f"Within Immature (n={len(stats['within_immature']['values'])} pairs): {stats['within_immature']['mean']:.3f}"
    )
    print(f"\nBetween Mature-Neurotic: {stats['between_mature_neurotic']['mean']:.3f}")
    print(f"Between Mature-Immature: {stats['between_mature_immature']['mean']:.3f}")
    print(
        f"Between Neurotic-Immature: {stats['between_neurotic_immature']['mean']:.3f}"
    )
    print(f"\nOverall mean |cos|: {stats['overall_mean']:.3f}")

    # Save results
    output_dir = Path("/home/fqwqf/persona/defense_mechanism_analysis")
    output_dir.mkdir(exist_ok=True)

    results = {
        "model": model_name,
        "layer": layer,
        "mechanisms": mechanisms,
        "cosine_matrix": cos_matrix.tolist(),
        "statistics": {
            k: {"mean": float(v["mean"]), "values": [float(x) for x in v["values"]]}
            for k, v in stats.items()
            if k != "overall_mean"
        },
        "overall_mean": float(stats["overall_mean"]),
        "vaillant_hierarchy": VAILLANT_HIERARCHY,
    }

    with open(output_dir / "full_defense_orthogonality.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_dir / 'full_defense_orthogonality.json'}")

    # Create visualization
    plot_path = Path(
        "/home/fqwqf/persona/paper/figures/defense_mechanism_orthogonality_full.png"
    )
    plot_orthogonality_matrix(cos_matrix, mechanisms, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
