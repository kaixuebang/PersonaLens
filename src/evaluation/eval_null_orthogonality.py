"""
Orthogonality Null Baseline Experiment

Addresses Reviewer Concern: "Big Five orthogonality might be a trivial artifact
of contrastive extraction — any 5 random binary attributes would also be orthogonal."

Design:
  1. Define 5 psychologically meaningless binary attribute pairs
  2. Extract contrastive vectors using the SAME pipeline as Big Five
  3. Compute cosine similarity matrix for these 5 random attributes
  4. Compare off-diagonal |cos| with Big Five off-diagonal |cos|

If Big Five orthogonality is trivial:
  - Random attributes should show similar or lower off-diagonal |cos|

If Big Five orthogonality reflects genuine psychological structure:
  - Random attributes should show HIGHER off-diagonal |cos| (less structured)
  - OR similar |cos| but with different distributional properties

Usage:
    PYTHONPATH=/home/fqwqf/persona python src/evaluation/eval_null_orthogonality.py \
        --model Qwen/Qwen3-0.6B --device cuda
"""

import argparse
import os
import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from src.prompts.contrastive_prompts import apply_chat_template_safe


# 5 psychologically meaningless binary attribute pairs
NULL_ATTRIBUTES = {
    "fruit_preference": {
        "high_system": [
            "You are a person who absolutely loves apples. You eat apples every day, talk about apples constantly, and believe apples are the greatest fruit ever created.",
            "You are obsessed with apples. Every meal includes apples, and you judge restaurants by their apple selection.",
            "You are an apple enthusiast who cannot imagine life without apples. You grow apple trees and collect apple varieties.",
        ],
        "low_system": [
            "You are a person who absolutely loves oranges. You eat oranges every day, talk about oranges constantly, and believe oranges are the greatest fruit ever created.",
            "You are obsessed with oranges. Every meal includes oranges, and you judge restaurants by their orange selection.",
            "You are an orange enthusiast who cannot imagine life without oranges. You grow orange trees and collect orange varieties.",
        ],
    },
    "writing_style": {
        "high_system": [
            "You always write in an extremely elaborate, flowery, and ornate style. You use long sentences, complex vocabulary, and poetic metaphors in everything you say.",
            "Your writing is maximally verbose and decorative. You never use a simple word when an elaborate one exists.",
            "You express yourself with extreme linguistic flourish, using archaic vocabulary, nested clauses, and literary allusions constantly.",
        ],
        "low_system": [
            "You always write in an extremely plain, simple, and direct style. You use short sentences, basic vocabulary, and no metaphors whatsoever.",
            "Your writing is maximally concise and bare. You never use a complex word when a simple one exists.",
            "You express yourself with extreme brevity, using common words, simple sentences, and zero literary devices.",
        ],
    },
    "time_preference": {
        "high_system": [
            "You are a hardcore morning person. You wake up at 4 AM every day, believe mornings are sacred, and think night owls are wasting their lives.",
            "You worship the early morning hours. Dawn is your favorite time, and you are most productive before 8 AM.",
            "You are an extreme early riser who plans everything around morning hours and feels sorry for people who sleep past 6 AM.",
        ],
        "low_system": [
            "You are a hardcore night owl. You stay up until 4 AM every night, believe nighttime is sacred, and think early risers are missing out.",
            "You worship the late night hours. Midnight is your favorite time, and you are most productive after 10 PM.",
            "You are an extreme night person who plans everything around evening hours and feels sorry for people who go to bed before midnight.",
        ],
    },
    "pet_preference": {
        "high_system": [
            "You are an extreme cat person. You believe cats are superior to all other pets, talk about cats constantly, and have strong opinions about cat breeds.",
            "You are obsessed with cats. You see the world through a cat-lover's lens and relate everything back to feline behavior.",
            "You are a devoted cat enthusiast who believes cats are the most intelligent and elegant creatures on earth.",
        ],
        "low_system": [
            "You are an extreme dog person. You believe dogs are superior to all other pets, talk about dogs constantly, and have strong opinions about dog breeds.",
            "You are obsessed with dogs. You see the world through a dog-lover's lens and relate everything back to canine behavior.",
            "You are a devoted dog enthusiast who believes dogs are the most loyal and wonderful creatures on earth.",
        ],
    },
    "season_preference": {
        "high_system": [
            "You are obsessed with summer. You believe summer is the only good season, hate cold weather, and plan your entire life around warm sunny days.",
            "You are a summer fanatic who lives for beach days, sunshine, and heat. Winter depresses you deeply.",
            "You worship summer and everything associated with it. You believe people who prefer winter are fundamentally misguided.",
        ],
        "low_system": [
            "You are obsessed with winter. You believe winter is the only good season, hate hot weather, and plan your entire life around cold snowy days.",
            "You are a winter fanatic who lives for snow days, cozy fires, and cold. Summer exhausts you deeply.",
            "You worship winter and everything associated with it. You believe people who prefer summer are fundamentally misguided.",
        ],
    },
}

# Shared neutral scenarios (same for all attributes)
NULL_SCENARIOS = [
    "Tell me about your typical day.",
    "What do you think about when you're alone?",
    "How do you make decisions?",
    "Describe your ideal vacation.",
    "What matters most to you in life?",
    "How do you handle disagreements?",
    "What's your approach to trying new things?",
    "How do you spend your free time?",
    "What do you value in friendships?",
    "How do you deal with stress?",
    "What's your philosophy on work-life balance?",
    "How do you approach learning something new?",
    "What role does creativity play in your life?",
    "How do you handle uncertainty?",
    "What's your perspective on change?",
    "How do you prioritize your goals?",
    "What do you think about taking risks?",
    "How do you recharge after a long week?",
    "What's your approach to problem-solving?",
    "How do you feel about routine?",
]


def collect_null_activations(model, tokenizer, device, n_scenarios=20):
    """Collect activations for all 5 null attributes."""
    import random

    all_activations = {}
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    for attr_name, attr_data in NULL_ATTRIBUTES.items():
        print(f"\n  Collecting activations for: {attr_name}")
        rng = random.Random(42 + hash(attr_name))

        pos_acts = {l: [] for l in range(n_layers)}
        neg_acts = {l: [] for l in range(n_layers)}

        scenarios = NULL_SCENARIOS[:n_scenarios]

        for scenario in tqdm(scenarios, desc=f"  {attr_name}"):
            high_sys = rng.choice(attr_data["high_system"])
            low_sys = rng.choice(attr_data["low_system"])

            for sys_prompt, acts_dict in [(high_sys, pos_acts), (low_sys, neg_acts)]:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": scenario},
                ]
                text = apply_chat_template_safe(
                    tokenizer, messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)

                hidden_states = []
                hooks = []

                def make_hook(storage):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            storage.append(output[0][:, -1, :].detach().cpu().numpy())
                        else:
                            storage.append(output[:, -1, :].detach().cpu().numpy())

                    return hook_fn

                layers = (
                    model.model.layers
                    if hasattr(model, "model") and hasattr(model.model, "layers")
                    else model.transformer.h
                )

                for i, layer in enumerate(layers):
                    hs = []
                    hidden_states.append(hs)
                    hooks.append(layer.register_forward_hook(make_hook(hs)))

                with torch.no_grad():
                    model(**inputs)

                for h in hooks:
                    h.remove()

                for l_idx, hs in enumerate(hidden_states):
                    if hs:
                        acts_dict[l_idx].append(hs[0][0])

        # Stack into arrays
        all_activations[attr_name] = {
            "pos": {l: np.array(v) for l, v in pos_acts.items() if v},
            "neg": {l: np.array(v) for l, v in neg_acts.items() if v},
        }

    return all_activations


def extract_null_vectors(all_activations, layer):
    """Extract mean-diff vectors for null attributes at given layer."""
    vectors = {}
    for attr_name, acts in all_activations.items():
        if layer in acts["pos"] and layer in acts["neg"]:
            diff = np.mean(acts["pos"][layer], axis=0) - np.mean(
                acts["neg"][layer], axis=0
            )
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                vectors[attr_name] = diff / norm
    return vectors


def compute_null_orthogonality(null_vectors, big5_vectors):
    """Compare orthogonality of null attributes vs Big Five."""
    # Null attribute similarity matrix
    null_names = list(null_vectors.keys())
    null_V = np.vstack([null_vectors[n] for n in null_names])
    null_sim = cosine_similarity(null_V)

    # Big Five similarity matrix
    b5_names = list(big5_vectors.keys())
    b5_V = np.vstack([big5_vectors[n] for n in b5_names])
    b5_sim = cosine_similarity(b5_V)

    # Off-diagonal statistics
    def off_diag_stats(sim_matrix):
        n = sim_matrix.shape[0]
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(abs(sim_matrix[i, j]))
        return {
            "mean": float(np.mean(off_diag)),
            "std": float(np.std(off_diag)),
            "max": float(np.max(off_diag)),
            "min": float(np.min(off_diag)),
            "values": [float(v) for v in off_diag],
        }

    null_stats = off_diag_stats(null_sim)
    b5_stats = off_diag_stats(b5_sim)

    return {
        "null_attributes": {
            "names": null_names,
            "similarity_matrix": null_sim.tolist(),
            "off_diagonal": null_stats,
        },
        "big_five": {
            "names": b5_names,
            "similarity_matrix": b5_sim.tolist(),
            "off_diagonal": b5_stats,
        },
        "comparison": {
            "null_mean_off_diag": null_stats["mean"],
            "b5_mean_off_diag": b5_stats["mean"],
            "ratio": null_stats["mean"] / (b5_stats["mean"] + 1e-10),
            "interpretation": (
                "NULL_HIGHER"
                if null_stats["mean"] > b5_stats["mean"] * 1.2
                else "SIMILAR"
                if abs(null_stats["mean"] - b5_stats["mean"]) < 0.05
                else "B5_HIGHER"
            ),
        },
    }


def plot_null_comparison(results, output_path):
    """Plot side-by-side comparison of null vs Big Five orthogonality."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Left: Null attribute similarity matrix
    null_sim = np.array(results["null_attributes"]["similarity_matrix"])
    null_names = [n.replace("_", "\n") for n in results["null_attributes"]["names"]]
    ax = axes[0]
    im = ax.imshow(null_sim, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(null_names)))
    ax.set_yticks(range(len(null_names)))
    ax.set_xticklabels(null_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(null_names, fontsize=8)
    for i in range(len(null_names)):
        for j in range(len(null_names)):
            color = "white" if abs(null_sim[i, j]) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{null_sim[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    ax.set_title(
        "Null Attributes\n(Random Binary Pairs)", fontsize=11, fontweight="bold"
    )

    # Middle: Big Five similarity matrix
    b5_sim = np.array(results["big_five"]["similarity_matrix"])
    b5_names = [n.capitalize() for n in results["big_five"]["names"]]
    ax = axes[1]
    im = ax.imshow(b5_sim, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(b5_names)))
    ax.set_yticks(range(len(b5_names)))
    ax.set_xticklabels(b5_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(b5_names, fontsize=8)
    for i in range(len(b5_names)):
        for j in range(len(b5_names)):
            color = "white" if abs(b5_sim[i, j]) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{b5_sim[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    ax.set_title(
        "Big Five Traits\n(Psychological Constructs)", fontsize=11, fontweight="bold"
    )

    # Right: Bar comparison
    ax = axes[2]
    null_off = results["null_attributes"]["off_diagonal"]["values"]
    b5_off = results["big_five"]["off_diagonal"]["values"]

    positions = [0, 1]
    means = [np.mean(null_off), np.mean(b5_off)]
    stds = [np.std(null_off), np.std(b5_off)]
    colors = ["#FF9800", "#2196F3"]

    bars = ax.bar(
        positions,
        means,
        yerr=stds,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        capsize=5,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["Null\nAttributes", "Big Five\nTraits"], fontsize=10)
    ax.set_ylabel("Mean Off-Diagonal |cos|", fontsize=10)
    ax.set_title("Orthogonality Comparison", fontsize=11, fontweight="bold")

    # Scatter individual values
    for i, (vals, color) in enumerate(zip([null_off, b5_off], colors)):
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            [i] * len(vals) + jitter,
            vals,
            color=color,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.3,
            s=30,
            zorder=5,
        )

    # Annotate
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.colorbar(im, ax=axes[:2], shrink=0.8, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Null orthogonality baseline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer for vectors (default: auto)"
    )
    parser.add_argument("--output_dir", type=str, default="null_orthogonality_results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {args.device}")
    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()

    model_short = args.model.replace("/", "_")

    # Determine layer
    if args.layer is None:
        analysis_path = (
            f"persona_vectors/{model_short}/openness/analysis_v2_openness.json"
        )
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                analysis = json.load(f)
            args.layer = analysis.get(
                "best_layer_loso",
                analysis.get("best_layer_snr", model.config.num_hidden_layers // 2),
            )
        else:
            args.layer = model.config.num_hidden_layers // 2
    print(f"Using layer: {args.layer}")

    # Collect null attribute activations
    print(f"\n{'=' * 60}")
    print(f"Null Orthogonality Baseline Experiment")
    print(f"Model: {args.model} | Layer: {args.layer}")
    print(f"{'=' * 60}\n")

    print("Phase 1: Collecting null attribute activations...")
    null_activations = collect_null_activations(model, tokenizer, args.device)

    print("\nPhase 2: Extracting null vectors...")
    null_vectors = extract_null_vectors(null_activations, args.layer)
    print(f"  Extracted {len(null_vectors)} null attribute vectors")

    # Load Big Five vectors
    print("\nPhase 3: Loading Big Five vectors...")
    big5_traits = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    big5_vectors = {}
    for trait in big5_traits:
        vec_path = f"persona_vectors/{model_short}/{trait}/vectors/mean_diff_layer_{args.layer}.npy"
        if os.path.exists(vec_path):
            vec = np.load(vec_path)
            big5_vectors[trait] = vec / np.linalg.norm(vec)
            print(f"  Loaded: {trait}")
        else:
            print(f"  WARNING: Missing vector for {trait}")

    if len(big5_vectors) < 3:
        print("ERROR: Not enough Big Five vectors found. Run extraction first.")
        return

    # Compare
    print("\nPhase 4: Computing orthogonality comparison...")
    results = compute_null_orthogonality(null_vectors, big5_vectors)

    # Save
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "null_orthogonality.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Print summary
    comp = results["comparison"]
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Null attributes mean |cos|: {comp['null_mean_off_diag']:.4f}")
    print(f"  Big Five mean |cos|:        {comp['b5_mean_off_diag']:.4f}")
    print(f"  Ratio (null/B5):            {comp['ratio']:.2f}")
    print(f"  Interpretation:             {comp['interpretation']}")
    print(f"{'=' * 60}")

    # Plot
    fig_path = os.path.join(output_dir, "null_orthogonality.png")
    plot_null_comparison(results, fig_path)

    paper_fig_path = "paper/figures/null_orthogonality.png"
    os.makedirs("paper/figures", exist_ok=True)
    plot_null_comparison(results, paper_fig_path)

    print("\n✓ Null orthogonality baseline experiment complete!")


if __name__ == "__main__":
    main()
