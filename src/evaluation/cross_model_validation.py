"""
Phase 6: Cross-Model Transfer Validation

Tests whether the standardized extraction methodology produces
consistent results across different model architectures and scales.

This is the key contribution: a METHOD that works across models,
not just findings specific to one model.

Usage:
    python cross_model_validation.py --models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B --trait openness
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.stats import spearmanr


def load_analysis(persona_vectors_dir, model_name, trait_name):
    """Load analysis results for a model-trait combination."""
    model_short = model_name.replace("/", "_")
    analysis_file = os.path.join(
        persona_vectors_dir, model_short, trait_name, f"analysis_v2_{trait_name}.json"
    )
    if os.path.exists(analysis_file):
        with open(analysis_file) as f:
            return json.load(f)
    return None


def compare_layer_profiles(analyses, trait_name, output_dir):
    """
    Compare how different models encode the same trait across layers.

    Key question: Do different models show the same "encoding profile"
    (i.e., early/middle/late layer specialization)?
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#E91E63"]

    correlation_matrix = {}

    for i, (model_name, analysis) in enumerate(analyses.items()):
        layers = sorted([int(k) for k in analysis["layers"].keys()])
        accs = [analysis["layers"][str(l)]["loso_accuracy"] for l in layers]

        # Normalize layer indices to [0, 1] for cross-model comparison
        normalized_layers = [l / max(layers) for l in layers]

        ax.plot(normalized_layers, accs, "o-", color=colors[i % len(colors)],
                linewidth=2, markersize=4,
                label=f"{model_name} (best: L{analysis['best_layer_loso']}, "
                      f"acc: {analysis['best_loso_accuracy']:.3f})")

        correlation_matrix[model_name] = {
            "normalized_layers": normalized_layers,
            "probe_accs": accs,
        }

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer Position (0=first, 1=last)", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title(f"Cross-Model Encoding Profile: {trait_name}", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f"cross_model_{trait_name}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross-model comparison to {fig_path}")

    # Compute pairwise Spearman correlations of encoding profiles
    model_names = list(correlation_matrix.keys())
    if len(model_names) >= 2:
        print(f"\n  Encoding profile correlations (Spearman) for {trait_name}:")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                # Interpolate to same number of points if needed
                accs_i = correlation_matrix[model_names[i]]["probe_accs"]
                accs_j = correlation_matrix[model_names[j]]["probe_accs"]
                min_len = min(len(accs_i), len(accs_j))
                # Simple comparison using relative positions
                idx_i = np.linspace(0, len(accs_i) - 1, min_len).astype(int)
                idx_j = np.linspace(0, len(accs_j) - 1, min_len).astype(int)
                rho, pval = spearmanr(
                    [accs_i[k] for k in idx_i],
                    [accs_j[k] for k in idx_j]
                )
                print(f"    {model_names[i]} vs {model_names[j]}: "
                      f"ρ={rho:.3f}, p={pval:.4f}")

    return correlation_matrix


def compare_vector_geometry(persona_vectors_dir, model_names, trait_names, output_dir):
    """
    Compare the geometry of persona vectors across traits within each model.
    Key question: Are Big Five dimensions orthogonal in activation space?
    """
    for model_name in model_names:
        model_short = model_name.replace("/", "_")
        vectors = {}

        # Load best-layer vectors for each trait
        for trait_name in trait_names:
            analysis_file = os.path.join(
                persona_vectors_dir, model_short, trait_name,
                f"analysis_v2_{trait_name}.json"
            )
            if not os.path.exists(analysis_file):
                continue
            with open(analysis_file) as f:
                analysis = json.load(f)
            best_layer = analysis["best_layer_loso"]

            vec_file = os.path.join(
                persona_vectors_dir, model_short, trait_name,
                "vectors", f"mean_diff_layer_{best_layer}.npy"
            )
            if os.path.exists(vec_file):
                vectors[trait_name] = np.load(vec_file)

        if len(vectors) < 2:
            continue

        # Compute cosine similarity matrix
        trait_list = sorted(vectors.keys())
        n = len(trait_list)
        cos_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                v_i = vectors[trait_list[i]]
                v_j = vectors[trait_list[j]]
                cos_matrix[i, j] = np.dot(v_i, v_j) / (
                    np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-10
                )

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cos_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(trait_list, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(trait_list, fontsize=9)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cos_matrix[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(cos_matrix[i,j]) > 0.5 else "black")

        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title(f"Persona Vector Orthogonality: {model_name}", fontsize=13)

        fig_path = os.path.join(output_dir, f"orthogonality_{model_short}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n  Orthogonality matrix for {model_name}:")
        print(f"  Mean off-diagonal |cos|: "
              f"{np.mean(np.abs(cos_matrix[np.triu_indices(n, k=1)])):.3f}")
        print(f"  Saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-model validation of personality extraction")
    parser.add_argument("--persona_vectors_dir", type=str, default="persona_vectors",
                        help="Directory containing persona vectors for all models")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (auto-detected if None)")
    parser.add_argument("--trait", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="cross_model_results")
    args = parser.parse_args()

    # Auto-detect models
    if args.models:
        model_names = args.models.split(",")
    else:
        model_names = []
        if os.path.isdir(args.persona_vectors_dir):
            for d in os.listdir(args.persona_vectors_dir):
                if os.path.isdir(os.path.join(args.persona_vectors_dir, d)):
                    model_names.append(d.replace("_", "/", 1))
    print(f"Models: {model_names}")

    # Auto-detect traits
    all_traits = set()
    for model_name in model_names:
        model_short = model_name.replace("/", "_")
        model_dir = os.path.join(args.persona_vectors_dir, model_short)
        if os.path.isdir(model_dir):
            for d in os.listdir(model_dir):
                if os.path.isdir(os.path.join(model_dir, d)):
                    all_traits.add(d)

    if args.trait == "all":
        traits = sorted(all_traits)
    else:
        traits = [args.trait]

    print(f"Traits: {traits}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Cross-model layer profile comparison
    for trait_name in traits:
        analyses = {}
        for model_name in model_names:
            analysis = load_analysis(args.persona_vectors_dir, model_name, trait_name)
            if analysis:
                analyses[model_name] = analysis

        if len(analyses) >= 1:
            compare_layer_profiles(analyses, trait_name, args.output_dir)

    # 2. Intra-model vector orthogonality
    if len(traits) >= 2:
        compare_vector_geometry(args.persona_vectors_dir, model_names, traits, args.output_dir)

    print(f"\n✓ Cross-model validation complete! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
