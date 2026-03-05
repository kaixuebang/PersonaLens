"""
Cross-Model Experiment Runner for PersonaLens
Runs the complete pipeline on multiple models and generates comparative results.
"""
import argparse
import os
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics.pairwise import cosine_similarity

BIG_FIVE = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
DEFENSE_MECHS = ["humor", "rationalization", "projection"]
ALL_TRAITS = BIG_FIVE + DEFENSE_MECHS

def run_cmd(cmd, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  ✗ FAILED: {label}")
        return False
    print(f"  ✓ COMPLETED: {label}")
    return True


def collect_all_traits(model_name, traits):
    """Phase 1: Collect activations for all traits."""
    python = sys.executable
    for trait in traits:
        act_dir = f"activations/{model_name.replace('/', '_')}/{trait}"
        if os.path.isdir(act_dir) and len(os.listdir(act_dir)) > 0:
            print(f"  [SKIP] Activations already exist for {trait}")
            continue
        success = run_cmd(
            [python, "src/localization/collect_activations.py", "--model", model_name, "--trait", trait],
            f"Collect Activations: {model_name} / {trait}"
        )
        if not success:
            return False
    return True


def extract_all_vectors(model_name, traits):
    """Phase 2: Extract persona vectors (v2) for all traits."""
    python = sys.executable
    model_short = model_name.replace("/", "_")
    act_dir = f"activations/{model_short}"
    out_dir = f"persona_vectors_v2/{model_short}"
    
    success = run_cmd(
        [python, "src/extraction/extract_persona_vectors_v2.py",
         "--activations_dir", act_dir,
         "--output_dir", out_dir,
         "--trait", "all"],
        f"Extract Vectors (v2): {model_name}"
    )
    return success


def localize_circuits(model_name, traits, n_samples=5):
    """Phase 3: Causal localization."""
    python = sys.executable
    for trait in traits:
        run_cmd(
            [python, "src/localization/localize_circuits_v2.py",
             "--model", model_name, "--trait", trait, "--n_samples", str(n_samples)],
            f"Causal Localization: {model_name} / {trait}"
        )


def evaluate_steering(model_name, trait="openness"):
    """Phase 4: Steering evaluation."""
    python = sys.executable
    run_cmd(
        [python, "src/evaluation/evaluate_steering_final.py", "--model", model_name, "--trait", trait],
        f"Steering Evaluation: {model_name} / {trait}"
    )


def compute_orthogonality_for_model(model_name, layer, traits=ALL_TRAITS):
    """Compute orthogonality matrix at a given layer for a model."""
    model_short = model_name.replace("/", "_")
    base_dir = f"persona_vectors_v2/{model_short}"
    
    vectors = {}
    for trait in traits:
        vec_path = os.path.join(base_dir, trait, "vectors", f"mean_diff_layer_{layer}.npy")
        if os.path.exists(vec_path):
            v = np.load(vec_path)
            vectors[trait] = v / (np.linalg.norm(v) + 1e-10)
    
    return vectors


def find_common_layer(model_name):
    """Find a good common layer (middle of the model) from analysis files."""
    model_short = model_name.replace("/", "_")
    base_dir = f"persona_vectors_v2/{model_short}"
    
    # Try to read analysis file for any trait
    for trait in ALL_TRAITS:
        analysis_path = os.path.join(base_dir, trait, f"analysis_v2_{trait}.json")
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                data = json.load(f)
            n_layers = data.get("n_layers", 28)
            return n_layers // 2
    return 14  # default


def generate_cross_model_orthogonality(model_configs, output_dir="cross_model_results"):
    """Generate side-by-side orthogonality matrices for all models."""
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(model_configs)
    
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    all_model_stats = {}
    
    for idx, (model_name, layer) in enumerate(model_configs):
        vectors = compute_orthogonality_for_model(model_name, layer)
        found_traits = [t for t in ALL_TRAITS if t in vectors]
        
        if len(found_traits) < 2:
            print(f"  Warning: Only {len(found_traits)} vectors found for {model_name} at layer {layer}")
            axes[idx].text(0.5, 0.5, f"Insufficient data\n({len(found_traits)} traits)", 
                          ha="center", va="center", fontsize=12)
            axes[idx].set_title(model_name)
            continue
        
        V = np.vstack([vectors[t] for t in found_traits])
        sim = cosine_similarity(V)
        
        # Compute stats
        n = len(found_traits)
        off_diag = sim[np.triu_indices(n, k=1)]
        b5_traits = [t for t in found_traits if t in BIG_FIVE]
        b5_idx = [found_traits.index(t) for t in b5_traits]
        
        b5_off = []
        for i in range(len(b5_idx)):
            for j in range(i+1, len(b5_idx)):
                b5_off.append(abs(sim[b5_idx[i], b5_idx[j]]))
        
        all_model_stats[model_name] = {
            "layer": layer,
            "n_traits": n,
            "mean_off_diag_cos": float(np.mean(np.abs(off_diag))),
            "mean_b5_off_diag_cos": float(np.mean(b5_off)) if b5_off else None,
            "found_traits": found_traits,
        }
        
        # Plot
        ax = axes[idx]
        cax = ax.imshow(sim, cmap="RdBu", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
        labels = [t[:4].capitalize() for t in found_traits]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        
        for i in range(n):
            for j in range(n):
                color = "white" if abs(sim[i,j]) > 0.5 else "black"
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", color=color, fontsize=7)
        
        model_label = model_name.split("/")[-1]
        ax.set_title(f"{model_label}\n(Layer {layer}, |cos|={np.mean(np.abs(off_diag)):.3f})", fontsize=10)
    
    plt.suptitle("Cross-Model Persona Vector Orthogonality", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "fig_cross_model_orthogonality.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Copy to paper
    import shutil
    shutil.copy(out_path, "paper/fig_cross_model_orthogonality.png")
    print(f"  Saved to {out_path} and paper/")
    
    # Save stats
    with open(os.path.join(output_dir, "cross_model_stats.json"), "w") as f:
        json.dump(all_model_stats, f, indent=2)
    
    return all_model_stats


def generate_cross_model_layer_profiles(model_configs, output_dir="cross_model_results"):
    """Compare layer-wise encoding profiles across models."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    
    for trait in BIG_FIVE[:3]:  # openness, conscientiousness, extraversion
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (model_name, _) in enumerate(model_configs):
            model_short = model_name.replace("/", "_")
            analysis_path = f"persona_vectors_v2/{model_short}/{trait}/analysis_v2_{trait}.json"
            
            if not os.path.exists(analysis_path):
                continue
            
            with open(analysis_path) as f:
                data = json.load(f)
            
            layers = sorted([int(k) for k in data["layers"].keys()])
            n_layers = len(layers)
            norm_layers = [l / max(layers) for l in layers]
            
            loso = [data["layers"][str(l)]["loso_accuracy"] for l in layers]
            norms = [data["layers"][str(l)]["rms_normalized_diff_norm"] for l in layers]
            ds = [data["layers"][str(l)]["cohens_d"] for l in layers]
            
            label = model_name.split("/")[-1]
            c = colors[idx % len(colors)]
            
            axes[0].plot(norm_layers, loso, "o-", color=c, linewidth=1.5, markersize=3, label=label, alpha=0.8)
            axes[1].plot(norm_layers, norms, "o-", color=c, linewidth=1.5, markersize=3, label=label, alpha=0.8)
            axes[2].plot(norm_layers, ds, "o-", color=c, linewidth=1.5, markersize=3, label=label, alpha=0.8)
        
        axes[0].set_title(f"LOSO Probe Accuracy — {trait.capitalize()}")
        axes[0].set_xlabel("Normalized Layer Position"); axes[0].set_ylabel("Accuracy")
        axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title(f"RMS-Normalized Diff Norm — {trait.capitalize()}")
        axes[1].set_xlabel("Normalized Layer Position"); axes[1].set_ylabel("Norm / RMS")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title(f"Cohen's d — {trait.capitalize()}")
        axes[2].set_xlabel("Normalized Layer Position"); axes[2].set_ylabel("Effect Size")
        axes[2].axhline(y=0.8, color="red", linestyle=":", alpha=0.3)
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"cross_model_profile_{trait}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved cross-model profile for {trait} to {out_path}")
    
    # Copy best one to paper
    import shutil
    shutil.copy(
        os.path.join(output_dir, "cross_model_profile_openness.png"),
        "paper/fig_cross_model_profile.png"
    )


def main():
    parser = argparse.ArgumentParser(description="Cross-Model PersonaLens Experiments")
    parser.add_argument("--models", type=str, 
                        default="Qwen/Qwen3-0.6B,Qwen/Qwen2.5-0.5B-Instruct",
                        help="Comma-separated model names")
    parser.add_argument("--traits", type=str, default="all",
                        help="'big5', 'defense', or 'all'")
    parser.add_argument("--skip_collect", action="store_true")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_localize", action="store_true")
    parser.add_argument("--skip_steer", action="store_true")
    parser.add_argument("--only_visualize", action="store_true",
                        help="Skip all computation, only generate cross-model visualizations")
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    
    if args.traits == "big5":
        traits = BIG_FIVE
    elif args.traits == "defense":
        traits = DEFENSE_MECHS
    else:
        traits = ALL_TRAITS
    
    print(f"\n{'#'*60}")
    print(f"  PersonaLens Cross-Model Experiments")
    print(f"  Models: {models}")
    print(f"  Traits: {traits}")
    print(f"{'#'*60}")
    
    if not args.only_visualize:
        for model in models:
            print(f"\n{'='*60}")
            print(f"  Processing: {model}")
            print(f"{'='*60}")
            
            # Phase 1: Collect
            if not args.skip_collect:
                collect_all_traits(model, traits)
            
            # Phase 2: Extract
            if not args.skip_extract:
                extract_all_vectors(model, traits)
            
            # Phase 3: Localize (only openness for speed)
            if not args.skip_localize:
                localize_circuits(model, ["openness"], n_samples=5)
            
            # Phase 4: Steer (only openness for speed)
            if not args.skip_steer:
                evaluate_steering(model, "openness")
    
    # Phase 5: Cross-model visualization
    print(f"\n{'='*60}")
    print(f"  Generating Cross-Model Visualizations")
    print(f"{'='*60}")
    
    model_configs = []
    for model in models:
        layer = find_common_layer(model)
        model_configs.append((model, layer))
        print(f"  {model}: common layer = {layer}")
    
    stats = generate_cross_model_orthogonality(model_configs)
    generate_cross_model_layer_profiles(model_configs)
    
    print(f"\n{'#'*60}")
    print(f"  ✓ CROSS-MODEL EXPERIMENTS COMPLETE")
    print(f"{'#'*60}")
    
    for model, s in stats.items():
        print(f"  {model}:")
        print(f"    Layer: {s['layer']}, Traits: {s['n_traits']}")
        print(f"    Mean |cos| (all): {s['mean_off_diag_cos']:.3f}")
        if s['mean_b5_off_diag_cos'] is not None:
            print(f"    Mean |cos| (B5):  {s['mean_b5_off_diag_cos']:.3f}")


if __name__ == "__main__":
    main()
