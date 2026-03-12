"""
Compute 9-defense-mechanism orthogonality matrix for multiple models.
Enables cross-model validation of Vaillant hierarchy in activation space.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

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

MECHANISM_ORDER = [
    "humor",
    "sublimation",
    "rationalization",
    "intellectualization",
    "displacement",
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
    base_path = Path(
        f"/home/fqwqf/persona/persona_vectors/{model_name}/{mechanism}/vectors"
    )
    vector_file = base_path / f"mean_diff_layer_{layer}.npy"
    if not vector_file.exists():
        raise FileNotFoundError(f"Vector not found: {vector_file}")
    vector = np.load(vector_file)
    return vector / np.linalg.norm(vector)


def compute_orthogonality_matrix(model_name, layer, mechanisms):
    n = len(mechanisms)
    cos_matrix = np.zeros((n, n))
    vectors = {}
    for mech in mechanisms:
        try:
            vectors[mech] = load_persona_vector(model_name, mech, layer)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            return None
    for i, mi in enumerate(mechanisms):
        for j, mj in enumerate(mechanisms):
            if mi in vectors and mj in vectors:
                cos_matrix[i, j] = np.dot(vectors[mi], vectors[mj])
    return cos_matrix


def compute_statistics(cos_matrix, mechanisms):
    n = len(mechanisms)
    mature_idx = [
        i for i, m in enumerate(mechanisms) if VAILLANT_HIERARCHY[m] == "mature"
    ]
    neurotic_idx = [
        i for i, m in enumerate(mechanisms) if VAILLANT_HIERARCHY[m] == "neurotic"
    ]
    immature_idx = [
        i for i, m in enumerate(mechanisms) if VAILLANT_HIERARCHY[m] == "immature"
    ]

    def off_diag(indices):
        return [np.abs(cos_matrix[i, j]) for i in indices for j in indices if i < j]

    def between(idx1, idx2):
        return [np.abs(cos_matrix[i, j]) for i in idx1 for j in idx2]

    return {
        "within_mature": {
            "mean": np.mean(off_diag(mature_idx)) if len(mature_idx) > 1 else 0,
            "values": off_diag(mature_idx),
        },
        "within_neurotic": {
            "mean": np.mean(off_diag(neurotic_idx)) if len(neurotic_idx) > 1 else 0,
            "values": off_diag(neurotic_idx),
        },
        "within_immature": {
            "mean": np.mean(off_diag(immature_idx)) if len(immature_idx) > 1 else 0,
            "values": off_diag(immature_idx),
        },
        "between_mature_neurotic": {
            "mean": np.mean(between(mature_idx, neurotic_idx)),
            "values": between(mature_idx, neurotic_idx),
        },
        "between_mature_immature": {
            "mean": np.mean(between(mature_idx, immature_idx)),
            "values": between(mature_idx, immature_idx),
        },
        "between_neurotic_immature": {
            "mean": np.mean(between(neurotic_idx, immature_idx)),
            "values": between(neurotic_idx, immature_idx),
        },
        "overall_mean": float(np.mean(np.abs(cos_matrix[np.triu_indices(n, k=1)]))),
    }


def plot_matrix(cos_matrix, mechanisms, model_name, layer, save_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    labels = []
    for mech in mechanisms:
        h = VAILLANT_HIERARCHY[mech]
        marker = {"mature": "\u2605", "neurotic": "\u25c6", "immature": "\u25cf"}[h]
        labels.append(f"{marker} {MECHANISM_LABELS[mech]}")

    im = ax.imshow(np.abs(cos_matrix), cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(mechanisms)))
    ax.set_yticks(np.arange(len(mechanisms)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Cosine Similarity|", rotation=270, labelpad=20, fontsize=11)

    for i in range(len(mechanisms)):
        for j in range(len(mechanisms)):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{np.abs(cos_matrix[i, j]):.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    ax.axhline(y=1.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.axvline(x=1.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.axhline(y=4.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.axvline(x=4.5, color="black", linewidth=2, linestyle="--", alpha=0.5)

    display_name = model_name.replace("Qwen_", "Qwen/").replace("unsloth_", "")
    ax.set_title(
        f"Defense Mechanism Orthogonality Matrix ({display_name}, Layer {layer})\nOrganized by Vaillant Hierarchy",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    configs = [
        {"model": "Qwen_Qwen3-0.6B", "layer": 14},  # 28 layers, 50% depth
    ]

    output_dir = Path("/home/fqwqf/persona/defense_mechanism_analysis")
    output_dir.mkdir(exist_ok=True)
    figures_dir = Path("/home/fqwqf/persona/paper/figures")

    all_results = {}

    for cfg in configs:
        model = cfg["model"]
        layer = cfg["layer"]
        print(f"\n{'=' * 60}")
        print(f"Model: {model}, Layer: {layer}")
        print(f"{'=' * 60}")

        cos_matrix = compute_orthogonality_matrix(model, layer, MECHANISM_ORDER)
        if cos_matrix is None:
            print(f"  SKIPPED - missing vectors")
            continue

        stats = compute_statistics(cos_matrix, MECHANISM_ORDER)

        print(f"  Within Mature:   {stats['within_mature']['mean']:.3f}")
        print(f"  Within Neurotic: {stats['within_neurotic']['mean']:.3f}")
        print(f"  Within Immature: {stats['within_immature']['mean']:.3f}")
        print(f"  Mature-Immature: {stats['between_mature_immature']['mean']:.3f}")
        print(f"  Neurotic-Immature: {stats['between_neurotic_immature']['mean']:.3f}")
        print(f"  Overall mean:    {stats['overall_mean']:.3f}")
        ratio = stats["within_immature"]["mean"] / max(
            stats["within_mature"]["mean"], 1e-6
        )
        print(f"  Immature/Mature ratio: {ratio:.1f}x")

        results = {
            "model": model,
            "layer": layer,
            "mechanisms": MECHANISM_ORDER,
            "cosine_matrix": cos_matrix.tolist(),
            "statistics": {
                k: {
                    "mean": float(v["mean"]) if isinstance(v, dict) else float(v),
                    "values": [float(x) for x in v["values"]]
                    if isinstance(v, dict) and "values" in v
                    else [],
                }
                for k, v in stats.items()
                if k != "overall_mean"
            },
            "overall_mean": float(stats["overall_mean"]),
            "vaillant_hierarchy": VAILLANT_HIERARCHY,
            "immature_mature_ratio": float(ratio),
        }
        all_results[model] = results

        with open(
            output_dir / f"defense_orthogonality_{model}_L{layer}.json", "w"
        ) as f:
            json.dump(results, f, indent=2)

        plot_matrix(
            cos_matrix,
            MECHANISM_ORDER,
            model,
            layer,
            figures_dir / f"defense_mechanism_orthogonality_{model}.png",
        )

    # Save combined
    with open(output_dir / "defense_orthogonality_crossmodel.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(
        f"\nSaved combined results to {output_dir / 'defense_orthogonality_crossmodel.json'}"
    )


if __name__ == "__main__":
    main()
