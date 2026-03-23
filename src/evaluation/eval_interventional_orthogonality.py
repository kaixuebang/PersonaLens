"""
Interventional Orthogonality: Does Injecting Trait A Leave Trait B Unchanged?

Motivation (Belrose et al., 2023 / Park et al., 2023):
  Cosine similarity proves geometric near-orthogonality but NOT causal independence.
  True disentanglement requires showing that steering along trait A does not
  shift the activation-space representation of trait B.

Design:
  1. For each "source trait" A, compute the persona vector v_A.
  2. Inject alpha * v_A into a batch of neutral prompts: h -> h + alpha * v_A.
  3. Measure the CHANGE in probe score for every OTHER trait B using its linear
     probe w_B (already trained during extraction).
  4. A genuinely disentangled representation should satisfy:
       Delta_score_B ≈ 0  for all B ≠ A

  If Big Five traits are truly independent, injecting extraversion should
  NOT increase agreeableness scores — unlike injecting "not apples" (apple
  preference is geometrically near-orthogonal, but has no semantic guarantee
  of independence in this causal sense).

Results interpretation:
  - Mean |Delta cross-probe| ≪ self-probe change → disentangled (good)
  - Cross-probe change comparable to self → correlated representations

Usage:
    PYTHONPATH=/home/fqwqf/persona python src/evaluation/eval_interventional_orthogonality.py \\
        --model Qwen/Qwen3-0.6B --alpha 3.0
"""

import argparse
import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from src.prompts.contrastive_prompts import apply_chat_template_safe, BIG_FIVE_PROMPTS


BIG5_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

# Neutral evaluation prompts: no personality hint whatsoever
NEUTRAL_EVAL_PROMPTS = [
    "What would you have for breakfast today?",
    "Describe the last time you helped a stranger.",
    "How do you usually plan your schedule?",
    "Tell me about a recent conversation you enjoyed.",
    "What do you think of rainy days?",
    "How do you decide what to do when you have free time?",
    "What's a recent project you worked on?",
    "How do you feel when a plan changes unexpectedly?",
    "Describe your morning routine.",
    "What comes to mind when you hear the word 'adventure'?",
]


def get_hidden_state(model, tokenizer, device, text, layer):
    """Get hidden state at last-token position for a given layer."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    hidden = [None]
    layers = (
        model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model.transformer.h
    )

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hidden[0] = h[:, -1, :].detach().cpu().float().numpy()

    handle = layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return hidden[0][0]  # shape: (d,)


def collect_neutral_activations(model, tokenizer, device, layer, n_prompts=10):
    """Collect hidden states for neutral prompts (no system prompt)."""
    activations = []
    for prompt in NEUTRAL_EVAL_PROMPTS[:n_prompts]:
        messages = [{"role": "user", "content": prompt}]
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        h = get_hidden_state(model, tokenizer, device, text, layer)
        activations.append(h)
    return np.array(activations)  # (n_prompts, d)


def train_probe_from_activations(pos_acts, neg_acts):
    """Train a logistic regression probe (strong L2) on provided activations."""
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
    probe = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
    probe.fit(X, y)
    return probe


def probe_score(probe, h):
    """Return probability of POSITIVE (high-trait) class for activation h."""
    h_2d = h.reshape(1, -1)
    return float(probe.predict_proba(h_2d)[0, 1])


def load_activations_for_trait(activations_dir, trait, layer):
    """Load pre-saved activations for a trait."""
    pos_p = os.path.join(activations_dir, trait, f"layer_{layer}_pos.npy")
    neg_p = os.path.join(activations_dir, trait, f"layer_{layer}_neg.npy")
    if not os.path.exists(pos_p):
        pos_p = os.path.join(activations_dir, trait, f"pos_layer_{layer}.npy")
        neg_p = os.path.join(activations_dir, trait, f"neg_layer_{layer}.npy")
    if not os.path.exists(pos_p):
        return None, None
    return np.load(pos_p), np.load(neg_p)


def load_persona_vector(persona_vectors_dir, model_short, trait, layer):
    """Load a pre-saved mean-diff persona vector."""
    vec_path = os.path.join(
        persona_vectors_dir,
        model_short,
        trait,
        "vectors",
        f"mean_diff_layer_{layer}.npy",
    )
    if os.path.exists(vec_path):
        v = np.load(vec_path)
        return v / (np.linalg.norm(v) + 1e-12)
    return None


def main():
    parser = argparse.ArgumentParser(description="Interventional orthogonality test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--alpha", type=float, default=3.0, help="Injection strength for intervention"
    )
    parser.add_argument("--activations_dir", type=str, default="results/activations")
    parser.add_argument(
        "--persona_vectors_dir", type=str, default="results/persona_vectors"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/interventional_ortho_results"
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_short = args.model.replace("/", "_")
    activations_dir = os.path.join(args.activations_dir, model_short)
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    # --- Auto-detect best layer ---
    if args.layer is None:
        analysis_path = os.path.join(
            args.persona_vectors_dir,
            model_short,
            "openness",
            "analysis_v2_openness.json",
        )
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                analysis = json.load(f)
            args.layer = analysis.get(
                "best_layer_loso", analysis.get("best_layer_snr", 11)
            )
        else:
            args.layer = 11
    layer = args.layer

    print(f"\n{'=' * 60}")
    print(f"Interventional Orthogonality Experiment")
    print(f"Model: {args.model} | Layer: {layer} | alpha={args.alpha}")
    print(f"{'=' * 60}\n")

    # --- Load model for collecting activations on neutral prompts ---
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

    # --- Train one probe per trait from pre-saved activations ---
    print("\nTraining per-trait probes...")
    probes = {}
    persona_vectors = {}
    for trait in BIG5_TRAITS:
        pos, neg = load_activations_for_trait(activations_dir, trait, layer)
        if pos is None:
            print(f"  WARNING: no activations for {trait}, skipping")
            continue
        probe = train_probe_from_activations(pos, neg)
        probes[trait] = probe

        vec = load_persona_vector(args.persona_vectors_dir, model_short, trait, layer)
        if vec is None:
            # Compute on the fly
            diff = np.mean(pos, axis=0) - np.mean(neg, axis=0)
            norm = np.linalg.norm(diff)
            vec = diff / (norm + 1e-12)
        persona_vectors[trait] = vec
        print(f"  ✓ {trait}: probe trained, vector loaded")

    available_traits = list(probes.keys())
    if len(available_traits) < 3:
        print("ERROR: Not enough traits available.")
        return

    # --- Collect baseline neutral activations ---
    print("\nCollecting baseline neutral activations...")
    baseline_acts = collect_neutral_activations(
        model, tokenizer, args.device, layer, n_prompts=len(NEUTRAL_EVAL_PROMPTS)
    )
    print(f"  Baseline shape: {baseline_acts.shape}")

    # --- Baseline probe scores (no intervention) ---
    baseline_scores = {}
    for trait, probe in probes.items():
        scores = [probe_score(probe, h) for h in baseline_acts]
        baseline_scores[trait] = float(np.mean(scores))
    print("\nBaseline scores (no intervention):")
    for t, s in baseline_scores.items():
        print(f"  {t}: {s:.3f}")

    # --- Interventional matrix ---
    # results[source_trait][target_trait] = mean delta in probe score
    print(f"\nRunning interventions (alpha={args.alpha})...")
    results = {}
    for source_trait in tqdm(available_traits, desc="Source traits"):
        v_source = persona_vectors[source_trait].astype(np.float32)
        results[source_trait] = {}

        # Inject alpha * v_source into each neutral activation
        intervened_acts = baseline_acts + args.alpha * v_source[np.newaxis, :]

        for target_trait in available_traits:
            target_probe = probes[target_trait]
            intervened_scores = [probe_score(target_probe, h) for h in intervened_acts]
            delta = float(np.mean(intervened_scores)) - baseline_scores[target_trait]
            results[source_trait][target_trait] = {
                "delta": delta,
                "baseline": baseline_scores[target_trait],
                "post_intervention": float(np.mean(intervened_scores)),
            }

    # --- Compute summary statistics ---
    print("\n" + "=" * 60)
    print("INTERVENTIONAL ORTHOGONALITY MATRIX (Delta probe scores)")
    print("=" * 60)
    print(f"{'':20}", end="")
    for t in available_traits:
        print(f"  {t[:8]:>8}", end="")
    print()

    self_deltas = []
    cross_deltas = []
    delta_matrix = np.zeros((len(available_traits), len(available_traits)))

    for i, source in enumerate(available_traits):
        print(f"Inject {source[:18]:18}", end="")
        for j, target in enumerate(available_traits):
            delta = results[source][target]["delta"]
            delta_matrix[i, j] = delta
            print(f"  {delta:+8.3f}", end="")
            if source == target:
                self_deltas.append(abs(delta))
            else:
                cross_deltas.append(abs(delta))
        print()

    mean_self = float(np.mean(self_deltas)) if self_deltas else 0
    mean_cross = float(np.mean(cross_deltas)) if cross_deltas else 0
    disentanglement_ratio = mean_self / (mean_cross + 1e-12)

    print(f"\n  Mean |self-delta|:  {mean_self:.4f}")
    print(f"  Mean |cross-delta|: {mean_cross:.4f}")
    print(f"  Disentanglement ratio (self/cross): {disentanglement_ratio:.2f}x")
    if disentanglement_ratio > 3.0:
        print("  → CAUSAL DISENTANGLEMENT CONFIRMED: traits are causally independent.")
    elif disentanglement_ratio > 1.5:
        print("  → MODERATE INDEPENDENCE: cross-probe effects exist but are weaker.")
    else:
        print("  → CORRELATED REPRESENTATIONS: traits share causal structure.")
    print("=" * 60)

    # --- Visualise ---
    fig, ax = plt.subplots(figsize=(8, 6))
    trait_labels = [t.capitalize()[:5] for t in available_traits]
    im = ax.imshow(delta_matrix, cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(available_traits)))
    ax.set_yticks(range(len(available_traits)))
    ax.set_xticklabels([f"→{t}" for t in trait_labels], rotation=45, ha="right")
    ax.set_yticklabels([f"Inject {t}" for t in trait_labels])
    for i in range(len(available_traits)):
        for j in range(len(available_traits)):
            color = "white" if abs(delta_matrix[i, j]) > 0.15 else "black"
            ax.text(
                j,
                i,
                f"{delta_matrix[i, j]:+.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, label="Delta Probe Score")
    ax.set_title(
        f"Interventional Orthogonality Matrix\n"
        f"(α={args.alpha}, layer={layer})\n"
        f"Diagonal=self-effect, Off-diagonal=cross-contamination\n"
        f"Disentanglement ratio: {disentanglement_ratio:.2f}x",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "interventional_orthogonality.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    paper_fig = "paper/figures/interventional_orthogonality.png"
    os.makedirs("paper/figures", exist_ok=True)
    # Re-generate for paper
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(delta_matrix, cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(available_traits)))
    ax.set_yticks(range(len(available_traits)))
    ax.set_xticklabels([f"→{t}" for t in trait_labels], rotation=45, ha="right")
    ax.set_yticklabels([f"Inject {t}" for t in trait_labels])
    for i in range(len(available_traits)):
        for j in range(len(available_traits)):
            color = "white" if abs(delta_matrix[i, j]) > 0.15 else "black"
            ax.text(
                j,
                i,
                f"{delta_matrix[i, j]:+.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, label="Delta Probe Score")
    ax.set_title(
        f"Interventional Orthogonality (α={args.alpha}, Layer {layer})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(paper_fig, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Save JSON ---
    output = {
        "model": args.model,
        "layer": layer,
        "alpha": args.alpha,
        "baseline_scores": baseline_scores,
        "delta_matrix": {
            s: {t: results[s][t]["delta"] for t in available_traits}
            for s in available_traits
        },
        "summary": {
            "mean_self_delta": mean_self,
            "mean_cross_delta": mean_cross,
            "disentanglement_ratio": disentanglement_ratio,
            "interpretation": (
                "CAUSAL_DISENTANGLED"
                if disentanglement_ratio > 3
                else "MODERATE"
                if disentanglement_ratio > 1.5
                else "CORRELATED"
            ),
        },
    }
    json_path = os.path.join(output_dir, "interventional_orthogonality.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved: {json_path}")

    print("\n✓ Interventional orthogonality experiment complete!")


if __name__ == "__main__":
    main()
