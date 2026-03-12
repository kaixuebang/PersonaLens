"""
Bootstrap Confidence Intervals for All Key Results

Adds 95% CIs via bootstrapping (1000 resamples) to:
  - Table 1: OOD generalization cosine similarities
  - Table 2: Orthogonality off-diagonal |cos|
  - Paraphrase control accuracies
  - BFI-44 dose-response scores

Usage:
    PYTHONPATH=/home/fqwqf/persona python src/evaluation/eval_bootstrap_ci.py \
        --model Qwen/Qwen3-0.6B
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, ci_level=0.95, seed=42):
    """
    Compute bootstrap confidence interval for an arbitrary statistic.

    Args:
        data: Input data (array or list of arrays)
        statistic_fn: Function that computes the statistic from data
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level (0.95 = 95% CI)
        seed: Random seed for reproducibility

    Returns:
        point_estimate, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    point_estimate = statistic_fn(data)

    bootstrap_stats = []
    n = len(data) if isinstance(data, (list, np.ndarray)) else data.shape[0]

    for _ in range(n_bootstrap):
        if isinstance(data, list):
            indices = rng.choice(n, size=n, replace=True)
            boot_sample = [data[i] for i in indices]
        else:
            indices = rng.choice(n, size=n, replace=True)
            boot_sample = data[indices]
        bootstrap_stats.append(statistic_fn(boot_sample))

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return float(point_estimate), ci_lower, ci_upper


def bootstrap_ood_results(model_short):
    """Add CIs to OOD generalization results."""
    print("\n[1] OOD Generalization CIs...")

    # Load aggregated results
    ood_path = "ood_aggregated_results.json"
    if not os.path.exists(ood_path):
        print("  WARNING: ood_aggregated_results.json not found. Skipping.")
        return None

    with open(ood_path) as f:
        ood_data = json.load(f)

    # Find the correct model key
    model_key = None
    for key in ood_data:
        if model_short.replace("_", "/") in key or key.lower().startswith(
            model_short.split("_")[0].lower()
        ):
            model_key = key
            break

    if model_key is None:
        # Try broader matching
        for key in ood_data:
            if model_short.split("_")[-1].lower() in key.lower():
                model_key = key
                break

    if model_key is None:
        print(
            f"  WARNING: Model {model_short} not found in OOD results. Available: {list(ood_data.keys())}"
        )
        return None

    model_data = ood_data[model_key]
    results = {}

    for trait, trait_data in model_data.items():
        if "cosine_similarities" in trait_data:
            cos_sims = np.array(trait_data["cosine_similarities"])
            if len(cos_sims) > 1:
                point, ci_lo, ci_hi = bootstrap_ci(cos_sims, np.mean, n_bootstrap=1000)
                results[trait] = {
                    "mean": point,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n_samples": len(cos_sims),
                }
                print(f"  {trait}: {point:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
            else:
                val = float(cos_sims[0])
                results[trait] = {"mean": val, "ci_lower": val, "ci_upper": val, "note": "single split"}
                print(f"  {trait}: {val:.3f} (single split)")
        elif "cosine" in trait_data:
            val = float(trait_data["cosine"])
            results[trait] = {"mean": val, "ci_lower": val, "ci_upper": val, "note": "single split, CI not applicable"}
            print(f"  {trait}: {val:.3f} (single split, no CI)")

    return results


def bootstrap_paraphrase_results(model_short):
    """Add CIs to paraphrase control results."""
    print("\n[2] Paraphrase Control CIs...")

    para_dir = f"paraphrase_control_results/{model_short}"
    if not os.path.exists(para_dir):
        print(f"  WARNING: {para_dir} not found. Skipping.")
        return None

    results = {}
    traits = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    for trait in traits:
        # Try both naming patterns
        json_path = os.path.join(para_dir, f"paraphrase_control_{trait}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(para_dir, f"paraphrase_{trait}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Get the best layer's results
        summary = data.get("summary", {})
        best_layer = summary.get("best_layer", 11)
        layer_data = data.get("layers", {}).get(str(best_layer), {})

        # Extract accuracies from the best layer
        a2b = layer_data.get("train_A_test_B", None)
        b2a = layer_data.get("train_B_test_A", None)

        if a2b is not None:
            n_b = data.get("template_B_samples", 5) * 2  # *2 for pos+neg
            z = 1.96
            # Wilson interval for A->B
            denom_ab = 1 + z**2 / n_b
            center_ab = (a2b + z**2 / (2 * n_b)) / denom_ab
            margin_ab = z * np.sqrt(a2b * (1 - a2b) / n_b + z**2 / (4 * n_b**2)) / denom_ab
            results[f"{trait}_train_A_test_B"] = {
                "accuracy": float(a2b),
                "ci_lower": max(0, float(center_ab - margin_ab)),
                "ci_upper": min(1, float(center_ab + margin_ab)),
                "n_samples": n_b,
                "method": "wilson_interval",
            }

        if b2a is not None:
            n_a = data.get("template_A_samples", 20) * 2
            z = 1.96
            denom_ba = 1 + z**2 / n_a
            center_ba = (b2a + z**2 / (2 * n_a)) / denom_ba
            margin_ba = z * np.sqrt(b2a * (1 - b2a) / n_a + z**2 / (4 * n_a**2)) / denom_ba
            results[f"{trait}_train_B_test_A"] = {
                "accuracy": float(b2a),
                "ci_lower": max(0, float(center_ba - margin_ba)),
                "ci_upper": min(1, float(center_ba + margin_ba)),
                "n_samples": n_a,
                "method": "wilson_interval",
            }

        # Also compute mean across directions
        a2b_key = f"{trait}_train_A_test_B"
        b2a_key = f"{trait}_train_B_test_A"
        if a2b_key in results and b2a_key in results:
            mean_acc = (results[a2b_key]["accuracy"] + results[b2a_key]["accuracy"]) / 2
            results[f"{trait}_mean"] = {
                "accuracy": mean_acc,
                "ci_lower": min(results[a2b_key]["ci_lower"], results[b2a_key]["ci_lower"]),
                "ci_upper": max(results[a2b_key]["ci_upper"], results[b2a_key]["ci_upper"]),
            }
            print(f"  {trait}: {mean_acc:.3f} [{results[f'{trait}_mean']['ci_lower']:.3f}, {results[f'{trait}_mean']['ci_upper']:.3f}]")


    return results


def bootstrap_bfi_results(model_short):
    """Add CIs to BFI-44 dose-response results."""
    print("\n[3] BFI-44 Dose-Response CIs...")

    bfi_dir = f"bfi_results/{model_short}"
    if not os.path.exists(bfi_dir):
        print(f"  WARNING: {bfi_dir} not found. Skipping.")
        return None

    results = {}
    traits = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    for trait in traits:
        json_path = os.path.join(bfi_dir, f"bfi_self_report_{trait}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        trait_results = {}
        for alpha_str, alpha_data in data.get("results", {}).items():
            if "item_responses" in alpha_data:
                # Bootstrap over individual BFI items
                raw_items = alpha_data["item_responses"]
                # Handle both formats: list of floats or list of dicts
                if isinstance(raw_items[0], dict):
                    items = np.array([item.get("rating", 0) for item in raw_items], dtype=float)
                else:
                    items = np.array(raw_items, dtype=float)
                # Filter out zeros/nans
                items = items[items > 0]
                if len(items) >= 3:
                    point, ci_lo, ci_hi = bootstrap_ci(items, np.mean, n_bootstrap=1000)
                    trait_results[alpha_str] = {
                        "mean": point,
                        "ci_lower": ci_lo,
                        "ci_upper": ci_hi,
                        "n_items": len(items),
                    }
            elif "trait_score" in alpha_data:
                trait_results[alpha_str] = {
                    "mean": float(alpha_data["trait_score"]),
                    "note": "no item-level data for bootstrap",
                }

        if trait_results:
            results[trait] = trait_results
            # Print summary for extreme alphas
            for a in ["-8.0", "0.0", "8.0"]:
                if a in trait_results and "ci_lower" in trait_results[a]:
                    r = trait_results[a]
                    print(
                        f"  {trait} (α={a}): {r['mean']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
                    )

    return results


def bootstrap_orthogonality(model_short, layer=None):
    """Bootstrap CIs for orthogonality off-diagonal values."""
    print("\n[4] Orthogonality CIs...")

    base_dir = f"persona_vectors/{model_short}"
    big5 = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    # Auto-detect layer
    if layer is None:
        analysis_path = os.path.join(base_dir, "openness", "analysis_v2_openness.json")
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                analysis = json.load(f)
            layer = analysis.get("best_layer_loso", analysis.get("best_layer_snr", 14))
        else:
            layer = 14

    # Load all vectors
    vectors = {}
    for trait in big5:
        vec_path = os.path.join(
            base_dir, trait, "vectors", f"mean_diff_layer_{layer}.npy"
        )
        if os.path.exists(vec_path):
            vec = np.load(vec_path)
            vectors[trait] = vec / np.linalg.norm(vec)

    if len(vectors) < 3:
        print("  WARNING: Not enough vectors for orthogonality analysis")
        return None

    # Compute pairwise cosine similarities
    trait_names = list(vectors.keys())
    n = len(trait_names)
    pair_sims = []
    pair_names = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = abs(float(np.dot(vectors[trait_names[i]], vectors[trait_names[j]])))
            pair_sims.append(sim)
            pair_names.append(f"{trait_names[i]}-{trait_names[j]}")

    pair_sims = np.array(pair_sims)

    # Bootstrap the mean off-diagonal |cos|
    point, ci_lo, ci_hi = bootstrap_ci(pair_sims, np.mean, n_bootstrap=1000)

    results = {
        "layer": layer,
        "mean_off_diagonal": point,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_pairs": len(pair_sims),
        "pairs": {name: float(sim) for name, sim in zip(pair_names, pair_sims)},
    }

    print(f"  Mean |cos| (Big Five): {point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Individual pairs:")
    for name, sim in zip(pair_names, pair_sims):
        print(f"    {name}: {sim:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for all results")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="bootstrap_ci_results")
    args = parser.parse_args()

    model_short = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Bootstrap Confidence Intervals")
    print(f"Model: {args.model}")
    print(f"{'=' * 60}")

    all_results = {}

    # 1. OOD
    ood = bootstrap_ood_results(model_short)
    if ood:
        all_results["ood"] = ood

    # 2. Paraphrase
    para = bootstrap_paraphrase_results(model_short)
    if para:
        all_results["paraphrase"] = para

    # 3. BFI-44
    bfi = bootstrap_bfi_results(model_short)
    if bfi:
        all_results["bfi44"] = bfi

    # 4. Orthogonality
    ortho = bootstrap_orthogonality(model_short)
    if ortho:
        all_results["orthogonality"] = ortho

    # Save
    json_path = os.path.join(output_dir, "bootstrap_ci_all.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All CIs saved to: {json_path}")
    print(f"{'=' * 60}")
    print("\n✓ Bootstrap CI computation complete!")


if __name__ == "__main__":
    main()
