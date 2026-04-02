import argparse
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut


def load_activations(trait_dir):
    pos, neg = {}, {}
    for f in sorted(os.listdir(trait_dir)):
        if f.startswith("pos_layer_") and f.endswith(".npy"):
            idx = int(f.split("_")[2].split(".")[0])
            pos[idx] = np.load(os.path.join(trait_dir, f))
        elif f.startswith("neg_layer_") and f.endswith(".npy"):
            idx = int(f.split("_")[2].split(".")[0])
            neg[idx] = np.load(os.path.join(trait_dir, f))
    return pos, neg


def cohens_d_paired(pos_acts, neg_acts):
    diff_vec = np.mean(pos_acts, axis=0) - np.mean(neg_acts, axis=0)
    direction = diff_vec / (np.linalg.norm(diff_vec) + 1e-10)
    pos_proj = pos_acts @ direction
    neg_proj = neg_acts @ direction
    n1, n2 = len(pos_proj), len(neg_proj)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(pos_proj, ddof=1) + (n2 - 1) * np.var(neg_proj, ddof=1))
        / (n1 + n2 - 2)
    )
    return float((np.mean(pos_proj) - np.mean(neg_proj)) / (pooled_std + 1e-10))


def loso_accuracy(pos_acts, neg_acts):
    K = len(pos_acts)
    accs = []
    for i in range(K):
        train_pos = np.delete(pos_acts, i, axis=0)
        train_neg = np.delete(neg_acts, i, axis=0)
        test_pos = pos_acts[i : i + 1]
        test_neg = neg_acts[i : i + 1]
        X_train = np.concatenate([train_pos, train_neg])
        y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        X_test = np.concatenate([test_pos, test_neg])
        y_test = np.array([1.0, 0.0])
        clf = LogisticRegression(max_iter=2000, C=0.01, solver="lbfgs")
        clf.fit(X_train, y_train)
        accs.append(clf.score(X_test, y_test))
    return float(np.mean(accs))


def compare_token_positions(activations_base_dir, model_name, traits, output_path):
    model_short = model_name.replace("/", "_")
    positions = ["last", "mean", "penultimate"]
    results = {}

    for trait in traits:
        results[trait] = {}
        for pos in positions:
            pos_dir = os.path.join(activations_base_dir, f"{model_short}_{pos}", trait)
            if not os.path.isdir(pos_dir):
                print(f"  [SKIP] {trait} / {pos}: dir not found")
                continue
            pos_acts, neg_acts = load_activations(pos_dir)
            if not pos_acts:
                continue
            layer_indices = sorted(pos_acts.keys())
            best_d = max(
                cohens_d_paired(pos_acts[l], neg_acts[l]) for l in layer_indices
            )
            best_acc = max(
                loso_accuracy(pos_acts[l], neg_acts[l]) for l in layer_indices
            )
            results[trait][pos] = {"best_cohens_d": best_d, "best_loso_acc": best_acc}
            print(f"  {trait} / {pos}: d={best_d:.3f}, acc={best_acc:.3f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {output_path}")
    return results


def print_summary(results):
    positions = ["last", "mean", "penultimate"]
    print("\n=== TOKEN POSITION ABLATION SUMMARY ===")
    print(
        f"{'Trait':<25} {'last_d':>8} {'mean_d':>8} {'penu_d':>8} {'last_acc':>9} {'mean_acc':>9} {'penu_acc':>9}"
    )
    print("-" * 80)
    for trait, pos_data in results.items():
        row = f"{trait:<25}"
        for pos in positions:
            d = pos_data.get(pos, {}).get("best_cohens_d", float("nan"))
            row += f" {d:>8.3f}"
        for pos in positions:
            acc = pos_data.get(pos, {}).get("best_loso_acc", float("nan"))
            row += f" {acc:>9.3f}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--traits",
        type=str,
        default="openness,conscientiousness,extraversion,agreeableness,neuroticism",
    )
    parser.add_argument("--activations_dir", type=str, default="results/activations")
    parser.add_argument(
        "--output", type=str, default="results/analysis/token_position_ablation.json"
    )
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]
    results = compare_token_positions(
        args.activations_dir, args.model, traits, args.output
    )
    print_summary(results)


if __name__ == "__main__":
    main()
