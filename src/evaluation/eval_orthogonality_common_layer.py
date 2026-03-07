import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import argparse
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser(description="Wait, common-layer orthogonality test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layer", type=int, default=14)
    args = parser.parse_args()

    model_short = args.model.replace("/", "_")
    base_dir = f"persona_vectors/{model_short}"
    
    # We will look at both Big Five and defense mechanisms
    big_five = ["openness", "extraversion", "agreeableness", "conscientiousness", "neuroticism"]
    defenses = ["humor", "projection", "rationalization"]
    all_traits = big_five + defenses

    vectors = {}
    for trait in all_traits:
        vec_path = os.path.join(base_dir, trait, "vectors", f"mean_diff_layer_{args.layer}.npy")
        if os.path.exists(vec_path):
            vectors[trait] = np.load(vec_path)
            # Normalize vector
            vectors[trait] = vectors[trait] / np.linalg.norm(vectors[trait])
        else:
            print(f"Warning: vector not found for {trait} at layer {args.layer}")

    found_traits = [t for t in all_traits if t in vectors]
    if not found_traits:
        print("No vectors found. Check extracted paths.")
        return

    # Compute similarity matrix
    V = np.vstack([vectors[t] for t in found_traits])
    sim_matrix = cosine_similarity(V)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(sim_matrix, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(cax)

    # Ticks
    ax.set_xticks(range(len(found_traits)))
    ax.set_yticks(range(len(found_traits)))
    
    labels = [t.capitalize() for t in found_traits]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(f"Cross-Trait Cosine Similarity\n(Common Layer {args.layer})", fontsize=12)

    # Annotations
    for i in range(len(found_traits)):
        for j in range(len(found_traits)):
            val = sim_matrix[i, j]
            # determine text color based on background
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_dir = f"cross_model_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fig_orthogonality_common_L{args.layer}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    # Also copy to paper
    import shutil
    shutil.copy(out_path, "paper/fig_orthogonality_common.png")
    print(f"Saved orthogonality map to {out_path} and paper/fig_orthogonality_common.png")

if __name__ == "__main__":
    main()
