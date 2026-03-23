"""
Shuffle-Label (Permutation) Baseline for Big Five Orthogonality

Addresses Reviewer Concern: "The contrastive extraction pipeline itself might
produce orthogonal vectors regardless of whether the labels are meaningful."

Design:
  1. Load pre-collected Big Five activations (h+ and h-)
  2. SHUFFLE the labels: randomly assign pairs as "high" or "low" without
     regard to their actual trait condition. This destroys all semantic signal.
  3. Extract "shuffled" mean-difference vectors using the identical pipeline.
  4. Compute the cosine similarity matrix of the 5 shuffled vectors.
  5. Compare off-diagonal |cos| with the genuine Big Five vectors.

Prediction:
  - If orthogonality is a pipeline artifact: shuffled vectors should be equally
    orthogonal (similar or lower |cos|) with high variance.
  - If orthogonality reflects genuine data structure: shuffled vectors should
    collapse toward ZERO, since random pairing cancels the semantic direction.
    Their off-diagonal |cos| values will be noisy and inconsistent — there will
    be NO structural reason for them to be orthogonal to each other.

Usage:
    PYTHONPATH=/home/fqwqf/persona python src/evaluation/eval_shuffle_label_baseline.py \\
        --model Qwen/Qwen3-0.6B --n_permutations 100
"""

import argparse
import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


BIG5_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def load_activations(activations_dir, trait, layer):
    """Load pre-extracted activations for a trait at a given layer."""
    pos_path = os.path.join(activations_dir, trait, f"layer_{layer}_pos.npy")
    neg_path = os.path.join(activations_dir, trait, f"layer_{layer}_neg.npy")

    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        # Try v2 format
        pos_path = os.path.join(activations_dir, trait, f"pos_layer_{layer}.npy")
        neg_path = os.path.join(activations_dir, trait, f"neg_layer_{layer}.npy")

    if not os.path.exists(pos_path):
        return None, None

    pos = np.load(pos_path)
    neg = np.load(neg_path)
    return pos, neg


def extract_mean_diff_vector(pos_acts, neg_acts):
    """Compute the normalised mean-difference vector."""
    diff = np.mean(pos_acts, axis=0) - np.mean(neg_acts, axis=0)
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return None
    return diff / norm


def load_genuine_vectors(persona_vectors_dir, model_short, traits, layer):
    """Load pre-extracted genuine persona vectors."""
    vectors = {}
    for trait in traits:
        vec_path = os.path.join(
            persona_vectors_dir,
            model_short,
            trait,
            "vectors",
            f"mean_diff_layer_{layer}.npy",
        )
        if os.path.exists(vec_path):
            v = np.load(vec_path)
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                vectors[trait] = v / norm
    return vectors


def compute_off_diagonal_stats(sim_matrix):
    """Extract off-diagonal absolute cosine similarity values."""
    n = sim_matrix.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(abs(sim_matrix[i, j]))
    vals = np.array(vals)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "values": vals.tolist(),
    }


def run_shuffle_baseline(pos_acts_dict, neg_acts_dict, n_permutations=100, seed=42):
    """
    For each permutation:
      - Pool all pos and neg activations for each trait
      - Randomly reassign K samples as "high" vs "low"
      - Extract mean-diff vector for each trait
      - Compute off-diagonal |cos| of the resulting 5-vector set
    Returns the distribution of mean off-diagonal |cos| over permutations.
    """
    rng = np.random.RandomState(seed)
    traits = list(pos_acts_dict.keys())
    off_diag_means = []

    for perm_idx in range(n_permutations):
        shuffled_vectors = {}

        for trait in traits:
            pos = pos_acts_dict[trait]
            neg = neg_acts_dict[trait]
            K = min(len(pos), len(neg))

            # Pool both halves together
            pooled = np.concatenate([pos[:K], neg[:K]], axis=0)
            # Randomly select K samples as the "positive" class
            idx = rng.permutation(len(pooled))
            shuffled_pos = pooled[idx[:K]]
            shuffled_neg = pooled[idx[K : 2 * K]]

            vec = extract_mean_diff_vector(shuffled_pos, shuffled_neg)
            if vec is not None:
                shuffled_vectors[trait] = vec

        if len(shuffled_vectors) >= 3:
            trait_names = list(shuffled_vectors.keys())
            V = np.vstack([shuffled_vectors[t] for t in trait_names])
            sim = cosine_similarity(V)
            stats_perm = compute_off_diagonal_stats(sim)
            off_diag_means.append(stats_perm["mean"])

    return np.array(off_diag_means)


def plot_results(genuine_mean, shuffle_distribution, output_path):
    """Visualise the comparison between genuine orthogonality and the shuffle baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of shuffle distribution
    ax = axes[0]
    ax.hist(
        shuffle_distribution,
        bins=20,
        color="#FF9800",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Shuffled Labels",
    )
    ax.axvline(
        genuine_mean,
        color="#2196F3",
        linewidth=2.5,
        linestyle="--",
        label=f"Genuine Big Five\n(mean |cos|={genuine_mean:.4f})",
    )
    ax.axvline(
        np.mean(shuffle_distribution),
        color="#FF9800",
        linewidth=2,
        linestyle="-",
        label=f"Shuffle Mean\n({np.mean(shuffle_distribution):.4f})",
    )
    ax.set_xlabel("Mean Off-Diagonal |cos|", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Shuffle-Label Baseline:\nOrthogonality under Random Assignment",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    # Right: box comparison
    ax = axes[1]
    data_to_plot = [shuffle_distribution, [genuine_mean]]
    bp = ax.boxplot(
        data_to_plot,
        labels=["Shuffled\nLabels", "Genuine\nBig Five"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor("#FF9800")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#2196F3")
    bp["boxes"][1].set_alpha(0.7)
    ax.set_ylabel("Mean Off-Diagonal |cos|", fontsize=12)
    ax.set_title("Comparison: Genuine vs Shuffled", fontsize=12, fontweight="bold")

    # p-value using one-sample t-test vs the genuine value
    t_stat, p_val = stats.ttest_1samp(shuffle_distribution, genuine_mean)
    ax.text(
        0.5,
        0.02,
        f"t-test: t={t_stat:.2f}, p={p_val:.3f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="darkred",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Shuffle-label orthogonality baseline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer to use (default: auto)"
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=100,
        help="Number of random shuffles to run",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="results/activations",
        help="Root activations directory",
    )
    parser.add_argument(
        "--persona_vectors_dir",
        type=str,
        default="results/persona_vectors",
        help="Root persona vectors directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/shuffle_label_baseline_results"
    )
    args = parser.parse_args()

    model_short = args.model.replace("/", "_")
    activations_dir = os.path.join(args.activations_dir, model_short)
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    # --- Auto-detect best layer ---
    if args.layer is None:
        analysis_path = os.path.join(
            args.persona_vectors_dir,
            model_short,
            "openness",
            "analysis_v2_openness.json",
        )
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                analysis = json.load(f)
            args.layer = analysis.get(
                "best_layer_loso",
                analysis.get("best_layer_snr", 11),
            )
        else:
            args.layer = 11
    layer = args.layer
    print(f"Using layer: {layer}")

    # --- Load activations ---
    print(f"\n{'=' * 60}")
    print(f"Shuffle-Label Baseline Experiment")
    print(f"Model: {args.model} | Layer: {layer} | Permutations: {args.n_permutations}")
    print(f"{'=' * 60}\n")

    pos_acts_dict = {}
    neg_acts_dict = {}
    missing_traits = []

    for trait in BIG5_TRAITS:
        pos, neg = load_activations(activations_dir, trait, layer)
        if pos is not None and neg is not None:
            pos_acts_dict[trait] = pos
            neg_acts_dict[trait] = neg
            print(f"  Loaded {trait}: pos={pos.shape}, neg={neg.shape}")
        else:
            missing_traits.append(trait)
            print(f"  WARNING: Missing activations for {trait} at layer {layer}")

    if len(pos_acts_dict) < 3:
        print(
            "\nERROR: Not enough trait activations found. "
            "Run activation collection first:\n"
            f"  PYTHONPATH=. python src/extraction/extract_persona_vectors_v2.py "
            f"--activations_dir activations/{model_short} --trait all\n"
            "or:\n"
            f"  PYTHONPATH=. python scripts/run_pipeline.py --model {args.model} --trait big5"
        )
        return

    # --- Genuine vectors ---
    print("\nLoading genuine Big Five vectors...")
    genuine_vectors = load_genuine_vectors(
        args.persona_vectors_dir, model_short, list(pos_acts_dict.keys()), layer
    )
    if len(genuine_vectors) < 3:
        print("  WARNING: Using on-the-fly extraction for genuine vectors.")
        for trait, (pos, neg) in zip(
            pos_acts_dict.keys(), zip(pos_acts_dict.values(), neg_acts_dict.values())
        ):
            vec = extract_mean_diff_vector(pos, neg)
            if vec is not None:
                genuine_vectors[trait] = vec

    trait_names = list(genuine_vectors.keys())
    V_genuine = np.vstack([genuine_vectors[t] for t in trait_names])
    genuine_sim = cosine_similarity(V_genuine)
    genuine_stats = compute_off_diagonal_stats(genuine_sim)
    genuine_mean = genuine_stats["mean"]
    print(
        f"  Genuine Big Five mean |cos|: {genuine_mean:.4f} ± {genuine_stats['std']:.4f}"
    )

    # --- Run shuffle baseline ---
    print(f"\nRunning {args.n_permutations} shuffle-label permutations...")
    shuffle_dist = run_shuffle_baseline(
        pos_acts_dict, neg_acts_dict, n_permutations=args.n_permutations
    )
    shuffle_mean = float(np.mean(shuffle_dist))
    shuffle_std = float(np.std(shuffle_dist))
    print(f"  Shuffled mean |cos|: {shuffle_mean:.4f} ± {shuffle_std:.4f}")

    # Statistical test: is genuine_mean below the shuffle distribution?
    t_stat, p_val = stats.ttest_1samp(shuffle_dist, genuine_mean)
    percentile_rank = float(np.mean(shuffle_dist > genuine_mean)) * 100
    effect_size = (shuffle_mean - genuine_mean) / (shuffle_std + 1e-12)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Genuine Big Five mean |cos|:  {genuine_mean:.4f}")
    print(f"  Shuffle baseline mean |cos|:  {shuffle_mean:.4f} ± {shuffle_std:.4f}")
    print(f"  Effect size (Cohen d equiv.): {effect_size:.2f}")
    print(f"  t-test (shuffle vs genuine):  t={t_stat:.2f}, p={p_val:.4f}")
    print(f"  {percentile_rank:.1f}% of shuffled perms exceed genuine mean")
    print(f"  Interpretation: ", end="")
    if p_val < 0.05 and genuine_mean < shuffle_mean:
        print("✓ GENUINE STRUCTURE CONFIRMED — shuffled labels produce HIGHER |cos|,")
        print(
            "    confirming that Big Five orthogonality is a data property, not a pipeline artifact."
        )
    elif p_val >= 0.05:
        print(
            "⚠ No significant difference — cannot confirm genuine structure this way."
        )
    else:
        print("⚠ Unexpected: genuine vectors are MORE correlated than shuffled.")
    print(f"{'=' * 60}")

    # --- Save results ---
    results = {
        "model": args.model,
        "layer": layer,
        "n_permutations": args.n_permutations,
        "genuine_big5": {
            "mean_off_diagonal": genuine_mean,
            "std_off_diagonal": genuine_stats["std"],
            "pair_sims": dict(
                zip(
                    [
                        f"{trait_names[i]}-{trait_names[j]}"
                        for i in range(len(trait_names))
                        for j in range(i + 1, len(trait_names))
                    ],
                    genuine_stats["values"],
                )
            ),
        },
        "shuffle_baseline": {
            "mean": shuffle_mean,
            "std": shuffle_std,
            "distribution": shuffle_dist.tolist(),
        },
        "statistics": {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "effect_size_cohen_d": float(effect_size),
            "pct_shuffle_exceeds_genuine": float(percentile_rank),
        },
        "interpretation": (
            "GENUINE_STRUCTURE"
            if p_val < 0.05 and genuine_mean < shuffle_mean
            else "INCONCLUSIVE"
        ),
    }

    json_path = os.path.join(output_dir, "shuffle_label_baseline.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # --- Plot ---
    fig_path = os.path.join(output_dir, "shuffle_label_baseline.png")
    plot_results(genuine_mean, shuffle_dist, fig_path)

    # Also copy to paper figures
    paper_fig_path = "paper/figures/shuffle_label_baseline.png"
    os.makedirs("paper/figures", exist_ok=True)
    plot_results(genuine_mean, shuffle_dist, paper_fig_path)

    print("\n✓ Shuffle-label baseline experiment complete!")


if __name__ == "__main__":
    main()
