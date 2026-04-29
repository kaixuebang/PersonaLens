"""Fast MLP vs Linear probe: Can non-linear probes separate O from E?"""
import numpy as np, json, os, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

ACT = "activations"
MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct", 15),
    ("unsloth_llama-3-8B-Instruct", 14),
    ("mistralai_Mistral-7B-Instruct-v0.1", 8),
    ("TinyLlama_TinyLlama-1.1B-Chat-v1.0", 19),
    ("Qwen_Qwen2.5-7B-Instruct", 14),
    ("unsloth_Llama-3.2-1B-Instruct", 7),
    ("unsloth_gemma-2-2b-it", 13),
]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def load_vec(model, trait, layer):
    d = os.path.join(ACT, model, trait)
    p, n = os.path.join(d, f"pos_layer_{layer}.npy"), os.path.join(d, f"neg_layer_{layer}.npy")
    if not os.path.exists(p): return None, None
    return np.load(p), np.load(n)

def main():
    results = {}
    oe_data = []

    for model, layer in MODELS:
        print(f"\n{model} (L{layer}):")
        acts = {}
        for t in TRAITS:
            p, n = load_vec(model, t, layer)
            if p is not None: acts[t] = (p, n)
        if len(acts) < 5:
            print(f"  Skip: {len(acts)} traits"); continue

        cross = {}
        for tr, te in [("openness","extraversion"), ("conscientiousness","neuroticism"),
                       ("openness","conscientiousness"), ("extraversion","agreeableness")]:
            if tr not in acts or te not in acts: continue
            tr_pos, tr_neg = acts[tr]; te_pos, te_neg = acts[te]
            X_tr = np.vstack([tr_pos, tr_neg]); y_tr = np.array([1]*len(tr_pos)+[0]*len(tr_neg))
            X_te = np.vstack([te_pos, te_neg]); y_te = np.array([1]*len(te_pos)+[0]*len(te_neg))

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
            pca = PCA(n_components=32, random_state=42)
            X_tr_p = pca.fit_transform(X_tr_s); X_te_p = pca.transform(X_te_s)

            lin = LogisticRegression(max_iter=500, C=1.0, solver="liblinear")
            lin.fit(X_tr_s, y_tr); lin_acc = lin.score(X_te_s, y_te)

            mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
            mlp.fit(X_tr_p, y_tr); mlp_acc = mlp.score(X_te_p, y_te)

            key = f"{tr[:3]}->{te[:3]}"
            cross[key] = {"linear": round(lin_acc,4), "mlp": round(mlp_acc,4), "delta": round(mlp_acc-lin_acc,4)}
            lbl = f"  {key}: Lin={lin_acc:.3f} MLP={mlp_acc:.3f}"
            if tr == "openness" and te == "extraversion": lbl += " ← O-E"
            print(lbl)
            if tr == "openness" and te == "extraversion":
                oe_data.append({"model": model, "linear": lin_acc, "mlp": mlp_acc})

        results[model] = {"layer": layer, "cross": cross}

    print(f"\n{'='*60}\nO-E CROSS-PROBE: CAN MLP SEPARATE O FROM E?\n{'='*60}")
    print(f"{'Model':35s} {'Linear':>8s} {'MLP':>8s} {'Delta':>8s}")
    print("-"*65)
    for e in oe_data:
        d = e["mlp"] - e["linear"]
        print(f"{e['model']:35s} {e['linear']:>8.3f} {e['mlp']:>8.3f} {d:>+8.3f}")
    if oe_data:
        deltas = [e["mlp"] - e["linear"] for e in oe_data]
        print(f"\nMean Delta(MLP - Linear) on O-E: {np.mean(deltas):+.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/mlp_vs_linear_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")

if __name__ == "__main__":
    main()
