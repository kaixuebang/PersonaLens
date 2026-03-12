"""
Evaluate Out-of-Domain Generalization & Stability

Addressed Reviewer Question: 
"How stable are vectors across seeds and scenario subsampling? Do mean-diff/probe 
directions remain aligned when extracted on disjoint scenario sets?"

This script loads activations, splits them into disjoint scenario sets (e.g. 50/50 splits),
extracts persona vectors on each disjoint set, and computes the cosine similarity
between the vectors to measure stability.

Usage:
    python eval_ood_generalization.py --activations_dir activations/Qwen_Qwen3-0.6B --trait openness
"""

import argparse
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def load_activations(trait_dir):
    pos_acts, neg_acts = {}, {}
    for f in sorted(os.listdir(trait_dir)):
        if f.startswith("pos_layer_") and f.endswith(".npy"):
            layer_idx = int(f.split("_")[2].split(".")[0])
            pos_acts[layer_idx] = np.load(os.path.join(trait_dir, f))
        elif f.startswith("neg_layer_") and f.endswith(".npy"):
            layer_idx = int(f.split("_")[2].split(".")[0])
            neg_acts[layer_idx] = np.load(os.path.join(trait_dir, f))
    return pos_acts, neg_acts

def extract_vectors(p, n, C=0.01):
    diff = np.mean(p, axis=0) - np.mean(n, axis=0)
    norm = np.linalg.norm(diff)
    mean_diff_dir = diff / norm if norm > 0 else diff
    
    X = np.concatenate([p, n], axis=0)
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs")
    clf.fit(X, y)
    probe_dir = clf.coef_[0]
    probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-10)
    
    return mean_diff_dir, probe_dir

def evaluate_stability(trait_dir, trait_name, output_dir, n_splits=5):
    pos_acts, neg_acts = load_activations(trait_dir)
    layers = sorted(pos_acts.keys())
    
    n_samples = pos_acts[layers[0]].shape[0]
    if n_samples < 4:
        print(f"Not enough samples ({n_samples}) for {trait_name} to do split evaluation.")
        return
        
    print(f"\nEvaluating OOD Generalization for {trait_name} ({n_samples} total scenarios)")
    
    results = {"mean_diff_cosine": {}, "probe_cosine": {}, "cross_method_cosine": {}}
    
    for layer in layers:
        p = pos_acts[layer]
        n = neg_acts[layer]
        
        md_sims = []
        pr_sims = []
        cross_sims = []
        
        for split_i in range(n_splits):
            # Fixed random seed for OOD split reproducibility
            np.random.seed(42 + split_i)
            perm = np.random.permutation(n_samples)
            mid = n_samples // 2
            set1_idx = perm[:mid]
            set2_idx = perm[mid:]
            
            p1, n1 = p[set1_idx], n[set1_idx]
            p2, n2 = p[set2_idx], n[set2_idx]
            
            md1, pr1 = extract_vectors(p1, n1)
            md2, pr2 = extract_vectors(p2, n2)
            
            md_sims.append(float(np.dot(md1, md2)))
            pr_sims.append(float(np.dot(pr1, pr2)))
            
            # Cross-method within the same disjoint set (e.g. md1 vs pr1 is already done, 
            # let's do md1 vs pr2 to see if methods generalize across data sets)
            cross_sims.append(float(np.dot(md1, pr2)))

        results["mean_diff_cosine"][layer] = {"mean": np.mean(md_sims), "std": np.std(md_sims)}
        results["probe_cosine"][layer] = {"mean": np.mean(pr_sims), "std": np.std(pr_sims)}
        results["cross_method_cosine"][layer] = {"mean": np.mean(cross_sims), "std": np.std(cross_sims)}
        
    # Plotting
    os.makedirs(os.path.join(output_dir, trait_name), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    layers_arr = np.array(layers)
    md_means = np.array([results["mean_diff_cosine"][l]["mean"] for l in layers])
    pr_means = np.array([results["probe_cosine"][l]["mean"] for l in layers])
    cross_means = np.array([results["cross_method_cosine"][l]["mean"] for l in layers])
    
    ax.plot(layers_arr, md_means, "o-", label="Mean-Diff (Set A vs Set B)", color="#2196F3")
    ax.plot(layers_arr, pr_means, "s-", label="Linear Probe (Set A vs Set B)", color="#FF9800")
    ax.plot(layers_arr, cross_means, "^-", label="MD (Set A) vs Probe (Set B)", color="#4CAF50")
    
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(f"Vector Stability Across Disjoint Scenarios — {trait_name}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(output_dir, trait_name, "ood_stability.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    with open(os.path.join(output_dir, trait_name, "ood_stability.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Find best intermediate layer (middle 50% of layers) for OOD stability
    n_layers = len(layers)
    start_idx = n_layers // 4  # Skip bottom 25%
    end_idx = 3 * n_layers // 4  # Skip top 25%
    intermediate_layers = layers[start_idx:end_idx]
    
    # Find best layer by mean_diff_cosine in intermediate layers
    best_layer = max(intermediate_layers, key=lambda l: results["mean_diff_cosine"][l]["mean"])
    best_cosine = results["mean_diff_cosine"][best_layer]["mean"]
    
    print(f"  ✓ Best intermediate layer (L={best_layer}): MD alignment = {best_cosine:.3f}")
    print(f"  ✓ Saved results to {fig_path}")
    
    # Add summary to results JSON
    results["summary"] = {
        "best_layer": int(best_layer),
        "best_layer_cosine": float(best_cosine),
        "intermediate_layers_range": [int(intermediate_layers[0]), int(intermediate_layers[-1])]
    }
    print(f"  ✓ Saved results to {fig_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--trait", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="ood_results")
    args = parser.parse_args()
    
    traits = [d for d in os.listdir(args.activations_dir) if os.path.isdir(os.path.join(args.activations_dir, d))] if args.trait == "all" else [args.trait]
    
    model_short = os.path.basename(args.activations_dir.rstrip("/"))
    out_dir = os.path.join(args.output_dir, model_short)
    
    for t in traits:
        tdir = os.path.join(args.activations_dir, t)
        if os.path.isdir(tdir):
            evaluate_stability(tdir, t, out_dir)

if __name__ == "__main__":
    main()
