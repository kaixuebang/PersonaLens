"""
Experiment: MLP vs Linear probe for personality trait classification.
Goal: Test whether non-linear probes can separate O from E better than linear probes.
This directly addresses Limitation (1): can non-linear directions overcome O-E collapse?
"""
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ACTIVATIONS_DIR = "activations"
RESULTS_FILE = "results/mlp_vs_linear_probe_results.json"

MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen3-0.6B",
    "unsloth_llama-3-8B-Instruct",
    "unsloth_Llama-3.1-8B-Instruct",
    "unsloth_Llama-3.2-1B-Instruct",
    "mistralai_Mistral-7B-Instruct-v0.1",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
    "unsloth_gemma-2-2b-it",
]

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Best layers per model (from our experiments)
BEST_LAYERS = {
    "Qwen_Qwen2.5-0.5B-Instruct": 15,
    "Qwen_Qwen2.5-1.5B-Instruct": 14,
    "Qwen_Qwen2.5-7B-Instruct": 14,
    "Qwen_Qwen2.5-14B-Instruct": 24,
    "Qwen_Qwen3-0.6B": 14,
    "unsloth_llama-3-8B-Instruct": 14,
    "unsloth_Llama-3.1-8B-Instruct": 14,
    "unsloth_Llama-3.2-1B-Instruct": 7,
    "mistralai_Mistral-7B-Instruct-v0.1": 8,
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": 19,
    "unsloth_gemma-2-2b-it": 13,
}


def load_activations(model, trait, layer):
    act_dir = os.path.join(ACTIVATIONS_DIR, model, trait)
    pos_file = os.path.join(act_dir, f"pos_layer_{layer}.npy")
    neg_file = os.path.join(act_dir, f"neg_layer_{layer}.npy")
    if not os.path.exists(pos_file) or not os.path.exists(neg_file):
        return None, None
    return np.load(pos_file), np.load(neg_file)


def run_probe_experiment(pos, neg, probe_type="linear", cv=5):
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if probe_type == "linear":
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    elif probe_type == "mlp":
        # 2-layer MLP with hidden dim = input_dim // 2
        hidden = max(32, X.shape[1] // 2)
        clf = MLPClassifier(hidden_layer_sizes=(hidden, hidden // 2),
                           max_iter=500, early_stopping=True,
                           validation_fraction=0.15, random_state=42)
    
    cv_strategy = StratifiedKFold(n_splits=min(cv, min(len(pos), len(neg))),
                                   shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv_strategy, scoring="accuracy")
    return scores.mean(), scores.std()


def main():
    results = {}
    
    for model in MODELS:
        layer = BEST_LAYERS.get(model)
        if layer is None:
            continue
        
        print(f"\n{'='*60}")
        print(f"Model: {model} (layer {layer})")
        print(f"{'='*60}")
        
        # Load all trait activations
        trait_acts = {}
        for trait in TRAITS:
            pos, neg = load_activations(model, trait, layer)
            if pos is not None:
                trait_acts[trait] = (pos, neg)
        
        if len(trait_acts) < 2:
            print(f"  Skipping: only {len(trait_acts)} traits available")
            continue
        
        # 1. Within-trait linear vs MLP (how well does each probe detect each trait)
        print(f"\n  Within-triat probe accuracy:")
        within_results = {}
        for trait, (pos, neg) in trait_acts.items():
            lin_acc, lin_std = run_probe_experiment(pos, neg, "linear")
            mlp_acc, mlp_std = run_probe_experiment(pos, neg, "mlp")
            within_results[trait] = {
                "linear_acc": round(lin_acc, 4),
                "linear_std": round(lin_std, 4),
                "mlp_acc": round(mlp_acc, 4),
                "mlp_std": round(mlp_std, 4),
                "mlp_improvement": round(mlp_acc - lin_acc, 4),
            }
            print(f"    {trait:20s}: Linear={lin_acc:.3f}±{lin_std:.3f}  MLP={mlp_acc:.3f}±{mlp_std:.3f}  Δ={mlp_acc-lin_acc:+.3f}")
        
        # 2. Cross-triat probe (the key experiment for O-E collapse)
        # Train on trait A, test on trait B — does MLP separate O from E?
        print(f"\n  Cross-triat probe (train→test):")
        cross_results = {}
        
        for train_trait in TRAITS:
            for test_trait in TRAITS:
                if train_trait == test_trait:
                    continue
                
                if train_trait not in trait_acts or test_trait not in trait_acts:
                    continue
                
                train_pos, train_neg = trait_acts[train_trait]
                test_pos, test_neg = trait_acts[test_trait]
                
                # Train on train_trait
                X_train = np.vstack([train_pos, train_neg])
                y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
                
                # Test on test_trait
                X_test = np.vstack([test_pos, test_neg])
                y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                # Linear probe
                lin_clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
                lin_clf.fit(X_train_s, y_train)
                lin_cross_acc = lin_clf.score(X_test_s, y_test)
                
                # MLP probe
                hidden = max(32, X_train.shape[1] // 2)
                mlp_clf = MLPClassifier(hidden_layer_sizes=(hidden, hidden // 2),
                                       max_iter=500, early_stopping=True,
                                       validation_fraction=0.15, random_state=42)
                mlp_clf.fit(X_train_s, y_train)
                mlp_cross_acc = mlp_clf.score(X_test_s, y_test)
                
                pair = f"{train_trait[:3]}->{test_trait[:3]}"
                cross_results[pair] = {
                    "train": train_trait,
                    "test": test_trait,
                    "linear_cross_acc": round(lin_cross_acc, 4),
                    "mlp_cross_acc": round(mlp_cross_acc, 4),
                    "mlp_improvement": round(mlp_cross_acc - lin_cross_acc, 4),
                }
                
                # Only print key pairs: O-E and C-N
                if (train_trait == "openness" and test_trait == "extraversion") or \
                   (train_trait == "conscientiousness" and test_trait == "neuroticism"):
                    print(f"    {pair:12s}: Linear={lin_cross_acc:.3f}  MLP={mlp_cross_acc:.3f}  Δ={mlp_cross_acc-lin_cross_acc:+.3f}")
        
        # Summarize O-E specifically
        oe_key = "ope->ext"
        cn_key = "con->neu"
        oe_linear = cross_results.get(oe_key, {}).get("linear_cross_acc", None)
        oe_mlp = cross_results.get(oe_key, {}).get("mlp_cross_acc", None)
        cn_linear = cross_results.get(cn_key, {}).get("linear_cross_acc", None)
        cn_mlp = cross_results.get(cn_key, {}).get("mlp_cross_acc", None)
        
        results[model] = {
            "layer": layer,
            "within_trait": within_results,
            "cross_probe": cross_results,
            "oe_linear": oe_linear,
            "oe_mlp": oe_mlp,
            "cn_linear": cn_linear,
            "cn_mlp": cn_mlp,
        }
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY: O-E Cross-Probe (Linear vs MLP)")
    print(f"{'='*60}")
    print(f"{'Model':35s} {'O-E Lin':>10s} {'O-E MLP':>10s} {'Δ':>8s} | {'C-N Lin':>10s} {'C-N MLP':>10s} {'Δ':>8s}")
    print("-" * 100)
    
    oe_lin_improvements = []
    for model, res in results.items():
        oe_l = res.get("oe_linear")
        oe_m = res.get("oe_mlp")
        cn_l = res.get("cn_linear")
        cn_m = res.get("cn_mlp")
        oe_d = f"{oe_m - oe_l:+.3f}" if oe_l and oe_m else "N/A"
        cn_d = f"{cn_m - cn_l:+.3f}" if cn_l and cn_m else "N/A"
        if oe_l and oe_m:
            oe_lin_improvements.append(oe_m - oe_l)
        print(f"{model:35s} {oe_l or 'N/A':>10s} {oe_m or 'N/A':>10s} {oe_d:>8s} | {cn_l or 'N/A':>10s} {cn_m or 'N/A':>10s} {cn_d:>8s}")
    
    if oe_lin_improvements:
        print(f"\nMean MLP improvement on O-E: {np.mean(oe_lin_improvements):+.3f} ± {np.std(oe_lin_improvements):.3f}")
    
    # Save
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
