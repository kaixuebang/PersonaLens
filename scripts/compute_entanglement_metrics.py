"""
compute_entanglement_metrics.py — Independent Entanglement Metrics

Computes entanglement metrics from RAW ACTIVATIONS only, completely
independent of steering results. Breaks the circular reasoning in
earlier analysis.

Metrics:
1. RV Coefficient: matrix correlation between trait subspaces.
   High RV = shared variance structure = entangled.
   Unlike CCA, RV is well-defined when n << d (K=20, d=896+).
2. Subspace Overlap: principal angles between trait subspaces.
3. Effective Dimensionality: PCA-based measure.
4. Cross-Probe Accuracy: train on trait A, test on trait B.
   High cross-accuracy = shared discriminative features = entangled.

Usage:
    python compute_entanglement_metrics.py
    python compute_entanglement_metrics.py --models Qwen_Qwen3-0.6B
"""

import argparse
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.linalg import subspace_angles
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVATIONS_DIR = os.path.join(REPO_ROOT, "activations")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "entanglement_metrics")

BIG_FIVE = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

MODEL_NAMES = {
    "Qwen_Qwen2.5-0.5B-Instruct": "Q2.5-0.5B",
    "Qwen_Qwen3-0.6B": "Q3-0.6B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "TinyLlama",
    "unsloth_Llama-3.2-1B-Instruct": "Llama-3.2",
    "Qwen_Qwen2.5-1.5B-Instruct": "Q2.5-1.5B",
    "unsloth_gemma-2-2b-it": "Gemma-2B",
    "Qwen_Qwen2.5-7B-Instruct": "Q2.5-7B",
    "mistralai_Mistral-7B-Instruct-v0.1": "Mistral-7B",
    "unsloth_Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
}


def load_all_trait_activations(model_dir, layer_idx):
    activations = {}
    for trait in BIG_FIVE:
        trait_dir = os.path.join(model_dir, trait)
        if not os.path.exists(trait_dir):
            continue
        pos_file = os.path.join(trait_dir, f"pos_layer_{layer_idx}.npy")
        neg_file = os.path.join(trait_dir, f"neg_layer_{layer_idx}.npy")
        if os.path.exists(pos_file) and os.path.exists(neg_file):
            pos = np.load(pos_file)
            neg = np.load(neg_file)
            activations[trait] = {"pos": pos, "neg": neg, "diff": pos - neg}
    return activations


def compute_rv_coefficient(activations):
    """
    RV coefficient: matrix correlation between trait activation matrices.
    RV in [0, 1]. Unlike CCA, well-defined when n << d.
    """
    traits = list(activations.keys())
    results = {}

    for ta, tb in combinations(traits, 2):
        X_a = activations[ta]["diff"]
        X_b = activations[tb]["diff"]

        S_a = X_a @ X_a.T
        S_b = X_b @ X_b.T

        numerator = np.trace(S_a @ S_b)
        denominator = np.sqrt(np.trace(S_a @ S_a) * np.trace(S_b @ S_b))

        rv = numerator / denominator if denominator > 1e-10 else 0.0
        results[(ta, tb)] = {"rv": float(rv)}

    return results


def compute_subspace_angles(activations, n_components=5):
    traits = list(activations.keys())
    results = {}

    for ta, tb in combinations(traits, 2):
        X_a = activations[ta]["diff"]
        X_b = activations[tb]["diff"]

        nc = min(n_components, min(X_a.shape[0], X_a.shape[1]))
        pca_a = PCA(n_components=nc).fit(X_a)
        pca_b = PCA(n_components=nc).fit(X_b)

        try:
            angles = subspace_angles(pca_a.components_.T, pca_b.components_.T)
            angles_deg = np.degrees(angles)
            results[(ta, tb)] = {
                "mean_angle": float(np.mean(angles_deg)),
                "min_angle": float(np.min(angles_deg)),
                "all_angles": [float(a) for a in angles_deg],
            }
        except Exception as e:
            results[(ta, tb)] = {"mean_angle": float('nan'), "error": str(e)}

    return results


def compute_effective_dimensionality(activations, variance_threshold=0.95):
    all_diffs = np.vstack([acts["diff"] for acts in activations.values()])

    n_components = min(all_diffs.shape[0] - 1, all_diffs.shape[1])
    pca = PCA(n_components=n_components).fit(all_diffs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_for_threshold = int(np.searchsorted(cumvar, variance_threshold) + 1)

    per_trait_ed = {}
    for trait, acts in activations.items():
        d = acts["diff"]
        nc = min(d.shape[0] - 1, d.shape[1])
        pca_t = PCA(n_components=nc).fit(d)
        cv = np.cumsum(pca_t.explained_variance_ratio_)
        per_trait_ed[trait] = int(np.searchsorted(cv, variance_threshold) + 1)

    return {
        "combined_effective_dim": n_for_threshold,
        "total_dims": all_diffs.shape[1],
        "variance_ratio": float(n_for_threshold / all_diffs.shape[1]),
        "explained_variance_top5": pca.explained_variance_ratio_[:5].tolist(),
        "per_trait_effective_dim": per_trait_ed,
        "cumulative_variance_at_5": float(cumvar[min(4, len(cumvar)-1)]),
    }


def compute_cross_probe_accuracy(activations):
    """
    Strongest entanglement test: train probe on trait A, test on trait B.
    High cross-accuracy = shared discriminative features = entangled.
    """
    traits = list(activations.keys())
    results = {}

    for ta, tb in combinations(traits, 2):
        X_a = np.vstack([activations[ta]["pos"], activations[ta]["neg"]])
        y_a = np.array([1]*len(activations[ta]["pos"]) + [0]*len(activations[ta]["neg"]))
        X_b = np.vstack([activations[tb]["pos"], activations[tb]["neg"]])
        y_b = np.array([1]*len(activations[tb]["pos"]) + [0]*len(activations[tb]["neg"]))

        # A -> B
        sc1 = StandardScaler()
        X_a_s = sc1.fit_transform(X_a)
        X_b_s = sc1.transform(X_b)
        p1 = LogisticRegression(C=0.01, max_iter=1000).fit(X_a_s, y_a)
        acc_ab = accuracy_score(y_b, p1.predict(X_b_s))

        # B -> A
        sc2 = StandardScaler()
        X_b_s2 = sc2.fit_transform(X_b)
        X_a_s2 = sc2.transform(X_a)
        p2 = LogisticRegression(C=0.01, max_iter=1000).fit(X_b_s2, y_b)
        acc_ba = accuracy_score(y_a, p2.predict(X_a_s2))

        within_a = accuracy_score(y_a, p1.predict(X_a_s))

        results[(ta, tb)] = {
            "A->B_acc": float(acc_ab),
            "B->A_acc": float(acc_ba),
            "mean_cross_acc": float((acc_ab + acc_ba) / 2),
            "within_A_acc": float(within_a),
        }

    return results


def find_best_layer(model_dir):
    pv_dir = os.path.join(REPO_ROOT, "results", "persona_vectors",
                          os.path.basename(model_dir))
    if not os.path.exists(pv_dir):
        return None

    best_layer = None
    best_score = -1
    for trait in BIG_FIVE:
        summary_file = os.path.join(pv_dir, f"extraction_summary_{trait}.json")
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                data = json.load(f)
                for layer_key, layer_data in data.items():
                    if layer_key.startswith("layer_"):
                        rms_norm = layer_data.get("rms_normalized_diff_norm", 0)
                        if rms_norm > best_score:
                            best_score = rms_norm
                            best_layer = int(layer_key.split("_")[1])
    return best_layer


def analyze_model(model_name, model_dir, layer=None):
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    if layer is None:
        layer = find_best_layer(model_dir)
        if layer is None:
            traits = os.listdir(model_dir)
            if traits:
                n_layers = len([f for f in os.listdir(os.path.join(model_dir, traits[0]))
                               if f.startswith("pos_layer_")])
                layer = n_layers // 2
            else:
                print(f"  Skipping {model_name}: no activation data")
                return None

    print(f"  Using layer {layer}")

    activations = load_all_trait_activations(model_dir, layer)
    if len(activations) < 2:
        print(f"  Skipping {model_name}: only {len(activations)} traits available")
        return None

    print(f"  Loaded {len(activations)} traits, shape: {list(activations.values())[0]['pos'].shape}")

    results = {"model": model_name, "layer": layer, "n_traits": len(activations)}

    # 1. RV Coefficient
    print("  Computing RV coefficients...")
    rv_results = compute_rv_coefficient(activations)
    rv_values = [v["rv"] for v in rv_results.values()]
    results["rv"] = {
        "pairwise": {f"{k[0]}-{k[1]}": v for k, v in rv_results.items()},
        "mean_rv": float(np.mean(rv_values)),
        "max_rv": float(np.max(rv_values)),
    }
    print(f"    Mean RV: {results['rv']['mean_rv']:.4f}")

    # 2. Subspace Angles
    print("  Computing subspace angles...")
    angle_results = compute_subspace_angles(activations)
    angle_values = [v["mean_angle"] for v in angle_results.values()
                    if not np.isnan(v.get("mean_angle", float('nan')))]
    results["subspace_angles"] = {
        "pairwise": {f"{k[0]}-{k[1]}": v for k, v in angle_results.items()},
        "mean_angle": float(np.mean(angle_values)) if angle_values else float('nan'),
        "min_angle": float(np.min(angle_values)) if angle_values else float('nan'),
    }
    print(f"    Mean subspace angle: {results['subspace_angles']['mean_angle']:.2f} deg")

    # 3. Effective Dimensionality
    print("  Computing effective dimensionality...")
    ed_results = compute_effective_dimensionality(activations)
    results["effective_dim"] = ed_results
    print(f"    Effective dim: {ed_results['combined_effective_dim']}/{ed_results['total_dims']} ({ed_results['variance_ratio']:.3f})")

    # 4. Cross-Probe Accuracy
    print("  Computing cross-probe accuracy...")
    cp_results = compute_cross_probe_accuracy(activations)
    cp_values = [v["mean_cross_acc"] for v in cp_results.values()]
    results["cross_probe"] = {
        "pairwise": {f"{k[0]}-{k[1]}": v for k, v in cp_results.items()},
        "mean_cross_acc": float(np.mean(cp_values)),
        "max_cross_acc": float(np.max(cp_values)),
    }
    print(f"    Mean cross-probe: {results['cross_probe']['mean_cross_acc']:.4f}")

    # Aggregate: higher = more entangled
    rv_score = results["rv"]["mean_rv"]
    angle_score = 1.0 - (results["subspace_angles"]["mean_angle"] / 90.0)
    ed_score = 1.0 - ed_results["variance_ratio"]
    cp_score = max(0, results["cross_probe"]["mean_cross_acc"] - 0.5) * 2

    results["aggregate_entanglement"] = {
        "rv_score": float(rv_score),
        "angle_score": float(angle_score),
        "ed_score": float(ed_score),
        "cp_score": float(cp_score),
        "overall": float(np.mean([rv_score, angle_score, ed_score, cp_score])),
    }
    print(f"    Aggregate: {results['aggregate_entanglement']['overall']:.4f}")

    return results


def generate_cross_model_comparison(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    comparison = {}
    for r in all_results:
        if r is None:
            continue
        name = MODEL_NAMES.get(r["model"], r["model"])
        comparison[name] = {
            "mean_rv": r["rv"]["mean_rv"],
            "mean_subspace_angle": r["subspace_angles"]["mean_angle"],
            "effective_dim_ratio": r["effective_dim"]["variance_ratio"],
            "mean_cross_probe": r["cross_probe"]["mean_cross_acc"],
            "aggregate_entanglement": r["aggregate_entanglement"]["overall"],
        }

    with open(os.path.join(output_dir, "cross_model_entanglement.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 90)
    print("CROSS-MODEL ENTANGLEMENT COMPARISON (independent of steering)")
    print("=" * 90)
    print(f"{'Model':<15} {'RV':>8} {'Angle':>8} {'ED Ratio':>10} {'CrossPrb':>10} {'Aggreg':>8}")
    print("-" * 90)
    for name, data in sorted(comparison.items(), key=lambda x: x[1]["aggregate_entanglement"]):
        print(f"{name:<15} {data['mean_rv']:>8.4f} {data['mean_subspace_angle']:>8.2f} "
              f"{data['effective_dim_ratio']:>10.4f} {data['mean_cross_probe']:>10.4f} "
              f"{data['aggregate_entanglement']:>8.4f}")

    if len(all_results) > 1:
        # Cross-probe heatmap (most informative metric)
        models = []
        cp_matrices = []
        for r in all_results:
            if r is None:
                continue
            name = MODEL_NAMES.get(r["model"], r["model"])
            models.append(name)
            n_traits = len(BIG_FIVE)
            mat = np.eye(n_traits)
            for key, vals in r["cross_probe"]["pairwise"].items():
                parts = key.split("-")
                ta, tb = parts[0], parts[1]
                i = BIG_FIVE.index(ta) if ta in BIG_FIVE else -1
                j = BIG_FIVE.index(tb) if tb in BIG_FIVE else -1
                if i >= 0 and j >= 0:
                    mat[i, j] = vals["mean_cross_acc"]
                    mat[j, i] = vals["mean_cross_acc"]
            cp_matrices.append(mat)

        ncols = min(len(models), 5)
        nrows = (len(models) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (model, mat) in enumerate(zip(models, cp_matrices)):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            im = ax.imshow(mat, cmap='RdYlBu_r', vmin=0.3, vmax=1.0)
            ax.set_xticks(range(n_traits))
            ax.set_yticks(range(n_traits))
            ax.set_xticklabels([t[:3].title() for t in BIG_FIVE], rotation=45, fontsize=7)
            ax.set_yticklabels([t[:3].title() for t in BIG_FIVE], fontsize=7)
            ax.set_title(model, fontsize=9)
            for i in range(n_traits):
                for j in range(n_traits):
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=6,
                           color='white' if mat[i,j] > 0.75 else 'black')

        for idx in range(len(models), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Cross-Probe Accuracy')
        fig.suptitle('Cross-Trait Entanglement (Cross-Probe Accuracy)', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cross_probe_heatmaps.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_models = sorted(comparison.items(), key=lambda x: x[1]["aggregate_entanglement"])
        names = [m for m, _ in sorted_models]
        entvals = [d["aggregate_entanglement"] for _, d in sorted_models]
        colors = ['#2ecc71' if v < 0.3 else '#f39c12' if v < 0.5 else '#e74c3c' for v in entvals]
        ax.barh(names, entvals, color=colors)
        ax.set_xlabel('Aggregate Entanglement Score')
        ax.set_title('Independent Entanglement Score (from raw activations only)')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='High entanglement')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aggregate_entanglement.png"), dpi=150, bbox_inches='tight')
        plt.close()

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.models:
        model_dirs = [(m, os.path.join(ACTIVATIONS_DIR, m)) for m in args.models]
    else:
        model_dirs = []
        for d in sorted(os.listdir(ACTIVATIONS_DIR)):
            full_path = os.path.join(ACTIVATIONS_DIR, d)
            if os.path.isdir(full_path):
                has_traits = any(os.path.isdir(os.path.join(full_path, t)) for t in BIG_FIVE)
                if has_traits:
                    model_dirs.append((d, full_path))

    print(f"Found {len(model_dirs)} models with activations")

    all_results = []
    for model_name, model_dir in model_dirs:
        result = analyze_model(model_name, model_dir, layer=args.layer)
        if result is not None:
            out_file = os.path.join(args.output_dir, f"{model_name}_entanglement.json")
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            all_results.append(result)

    comparison = generate_cross_model_comparison(all_results, args.output_dir)

    print(f"\nResults saved to {args.output_dir}")
    print(f"Analyzed {len(all_results)} models")


if __name__ == "__main__":
    main()
