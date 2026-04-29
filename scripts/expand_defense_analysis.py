"""
Expand defense mechanism analysis to all models with available activations.
Computes Vaillant hierarchy clustering + cross-model consistency.
"""
import numpy as np, json, os
from itertools import combinations
from scipy import stats

ACTIVATIONS_DIR = "activations"
RESULTS_DIR = "results/defense_mechanism_expanded"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFENSE_MECHANISMS = ['humor', 'sublimation', 'rationalization', 'intellectualization',
                      'displacement', 'projection', 'denial', 'regression', 'reaction_formation']
MATURE = ['humor', 'sublimation']
NEUROTIC = ['rationalization', 'intellectualization', 'displacement']
IMMATURE = ['projection', 'denial', 'regression', 'reaction_formation']

# Best layers per model (from Big Five experiments)
BEST_LAYERS = {
    "Qwen_Qwen2.5-0.5B-Instruct": 15,
    "Qwen_Qwen3-0.6B": 14,
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": 19,
    "unsloth_gemma-2-2b-it": 13,
    "unsloth_Llama-3.2-1B-Instruct": 7,
}


def get_trait_vector(model, trait, layer):
    """Get mean difference vector for a trait."""
    act_dir = os.path.join(ACTIVATIONS_DIR, model, trait)
    pos_f = os.path.join(act_dir, f"pos_layer_{layer}.npy")
    neg_f = os.path.join(act_dir, f"neg_layer_{layer}.npy")
    if not os.path.exists(pos_f) or not os.path.exists(neg_f):
        return None
    pos = np.load(pos_f)
    neg = np.load(neg_f)
    vec = pos.mean(axis=0) - neg.mean(axis=0)
    return vec / np.linalg.norm(vec)


def compute_pairwise_cosine(vectors):
    """Compute pairwise cosine similarity matrix."""
    names = list(vectors.keys())
    n = len(names)
    cos_matrix = {}
    for i in range(n):
        for j in range(i+1, n):
            cos = abs(np.dot(vectors[names[i]], vectors[names[j]]))
            cos_matrix[f"{names[i]}-{names[j]}"] = round(float(cos), 4)
    return cos_matrix


def compute_hierarchy_stats(cos_matrix):
    """Compute within-group statistics for Vaillant hierarchy."""
    def within_group(group, all_cos):
        vals = []
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                key1 = f"{group[i]}-{group[j]}"
                key2 = f"{group[j]}-{group[i]}"
                if key1 in all_cos:
                    vals.append(all_cos[key1])
                elif key2 in all_cos:
                    vals.append(all_cos[key2])
        return vals
    
    mature_vals = within_group(MATURE, cos_matrix)
    neurotic_vals = within_group(NEUROTIC, cos_matrix)
    immature_vals = within_group(IMMATURE, cos_matrix)
    
    return {
        "within_mature": {"mean": round(float(np.mean(mature_vals)), 4), "values": [round(v,4) for v in mature_vals]},
        "within_neurotic": {"mean": round(float(np.mean(neurotic_vals)), 4), "values": [round(v,4) for v in neurotic_vals]},
        "within_immature": {"mean": round(float(np.mean(immature_vals)), 4), "values": [round(v,4) for v in immature_vals]},
        "ratio_immature_to_mature": round(float(np.mean(immature_vals) / max(np.mean(mature_vals), 0.001)), 1),
    }


def main():
    print(f"{'='*60}")
    print("EXPANDED DEFENSE MECHANISM ANALYSIS")
    print(f"{'='*60}")
    
    all_results = {}
    hierarchy_order = []  # Track ordering across models
    
    for model, layer in BEST_LAYERS.items():
        print(f"\n{model} (L{layer}):")
        
        vectors = {}
        for mech in DEFENSE_MECHANISMS:
            vec = get_trait_vector(model, mech, layer)
            if vec is not None:
                vectors[mech] = vec
        
        if len(vectors) < 5:
            print(f"  Skip: only {len(vectors)} mechanisms")
            continue
        
        print(f"  {len(vectors)} mechanisms loaded")
        
        # Pairwise cosine
        cos_matrix = compute_pairwise_cosine(vectors)
        
        # Hierarchy stats
        hier = compute_hierarchy_stats(cos_matrix)
        
        # Check ordering: Immature > Neurotic > Mature?
        immature_mean = hier["within_immature"]["mean"]
        neurotic_mean = hier["within_neurotic"]["mean"]
        mature_mean = hier["within_mature"]["mean"]
        
        correct_order = immature_mean > neurotic_mean > mature_mean
        hierarchy_order.append(correct_order)
        
        print(f"  Mature:    {mature_mean:.4f}")
        print(f"  Neurotic:  {neurotic_mean:.4f}")
        print(f"  Immature:  {immature_mean:.4f}")
        print(f"  Ratio:     {hier['ratio_immature_to_mature']}x")
        print(f"  Ordering:  {'✓ Immature>Neurotic>Mature' if correct_order else '✗ Violated'}")
        
        # Cross-mechanism with Big Five: do defense mechanisms share representations with personality?
        bf_traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        bf_vectors = {}
        for t in bf_traits:
            vec = get_trait_vector(model, t, layer)
            if vec is not None:
                bf_vectors[t] = vec
        
        bf_defense_cos = {}
        if bf_vectors:
            for bf_t, bf_v in bf_vectors.items():
                for dm, dm_v in vectors.items():
                    cos = abs(np.dot(bf_v, dm_v))
                    bf_defense_cos[f"{bf_t}-{dm}"] = round(float(cos), 4)
            
            # Max cross-framework similarity
            max_cos = max(bf_defense_cos.values())
            mean_cos = round(float(np.mean(list(bf_defense_cos.values()))), 4)
            print(f"  BF-Defense: max|cos|={max_cos:.4f}, mean={mean_cos:.4f}")
        
        all_results[model] = {
            "layer": layer,
            "n_mechanisms": len(vectors),
            "cosine_matrix": cos_matrix,
            "hierarchy": hier,
            "correct_order": correct_order,
            "bf_defense_cross_cos": bf_defense_cos if bf_vectors else {},
        }
    
    # Summary
    n_correct = sum(hierarchy_order)
    n_total = len(hierarchy_order)
    
    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    
    for model, res in all_results.items():
        h = res["hierarchy"]
        order = "✓" if res["correct_order"] else "✗"
        print(f"  {model:40s}  M={h['within_mature']['mean']:.3f}  N={h['within_neurotic']['mean']:.3f}  I={h['within_immature']['mean']:.3f}  {h['ratio_immature_to_mature']:>5.1f}x  {order}")
    
    print(f"\nVaillant hierarchy (Immature>Neurotic>Mature): {n_correct}/{n_total} models")
    
    if n_total > 0:
        # Statistical test: probability of all models showing same ordering by chance
        # P(correct order | random) = 1/6 for 3 items
        p_all_correct = (1/6) ** n_correct
        print(f"P(all {n_correct} correct by chance) = {p_all_correct:.2e}")
    
    # Save
    summary = {
        "models": all_results,
        "summary": {
            "n_models": n_total,
            "n_correct_order": n_correct,
            "p_chance": float((1/6)**n_correct) if n_correct > 0 else None,
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "expanded_defense_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to {RESULTS_DIR}/expanded_defense_analysis.json")


if __name__ == "__main__":
    main()
