"""
Relative Injection Strength Analysis

Motivation (Todd et al., 2023 / Appendix discussion):
  Raw collapse thresholds (alpha_max) vary wildly across models:
    TinyLlama collapses at |alpha|=1.0, Gemma-2 tolerates |alpha|=10.0+.
  This variation is confounded by differences in residual stream norms.
  A unit steering vector injected at the scale of the residual stream
  has wildly different relative magnitude when the stream's L2 norm
  varies from ~5 to ~500.

  Normalizing alpha by the mean residual stream norm at the injection layer
  yields a model-comparable "relative injection strength" (RIS):
      RIS = alpha / mean(||h_ell||)

If collapse occurs at the same RIS across architectures, we can attribute
the variation to residual stream scale rather than to inherent architecture
fragility — providing a clean, principled explanation.

Design:
  1. Collect hidden states for neutral prompts across all models.
  2. Compute mean L2 norm of residual stream at the injection layer.
  3. Compute relative steering thresholds RIS = alpha_max / mean_norm
  4. Report per-model and compare.

Usage:
    PYTHONPATH=/home/fqwqf/persona python src/evaluation/eval_relative_injection_strength.py
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
from src.prompts.contrastive_prompts import apply_chat_template_safe


# Known collapse thresholds from the steering experiments (Table 4 in paper)
MODEL_ALPHA_MAX = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.0,
    "Qwen/Qwen3-0.6B": 4.0,
    "unsloth/Llama-3.2-1B-Instruct": 3.0,
    "Qwen/Qwen2.5-0.5B-Instruct": 6.0,
    "unsloth/gemma-2-2b-it": 10.0,
}

MODEL_INJECTION_LAYERS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 11,
    "Qwen/Qwen3-0.6B": 14,
    "unsloth/Llama-3.2-1B-Instruct": 6,
    "Qwen/Qwen2.5-0.5B-Instruct": 12,
    "unsloth/gemma-2-2b-it": 8,
}

# Neutral probe prompts (no personality hint)
NEUTRAL_PROMPTS = [
    "What is your opinion on the weather today?",
    "Tell me about something you read recently.",
    "How would you describe your typical afternoon?",
    "What do you think about coffee?",
    "Describe something you look forward to.",
    "What is your approach to learning new things?",
    "How do you feel when plans change?",
    "Describe a pleasant experience from this week.",
]


def get_residual_norm_at_layer(model_name, layer=None, n_prompts=8, device="cpu"):
    """
    Load a model, run neutral prompts, and compute the mean L2 norm
    of the residual stream (hidden state) at the specified layer.
    Returns (mean_norm, std_norm, all_norms).
    """
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    if layer is None:
        layer = MODEL_INJECTION_LAYERS.get(model_name, n_layers // 2)

    print(f"    Collecting activations at layer {layer} (of {n_layers})...")

    layers_list = (
        model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model.transformer.h
    )

    norms = []
    all_hidden = []

    for prompt_text in NEUTRAL_PROMPTS[:n_prompts]:
        messages = [{"role": "user", "content": prompt_text}]
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        hidden_states = [None]

        def make_hook(storage):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                storage[0] = h.detach().cpu().float()

            return hook_fn

        handle = layers_list[layer].register_forward_hook(make_hook(hidden_states))
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        h = hidden_states[0]  # (1, seq_len, d)
        # Collect norms for ALL token positions (not just last)
        for tok_idx in range(h.shape[1]):
            n = float(h[0, tok_idx, :].norm().item())
            norms.append(n)
        # Also store last-token state
        all_hidden.append(h[0, -1, :].numpy())

    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    norms = np.array(norms)
    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "median_norm": float(np.median(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "layer_used": layer,
        "n_layers": n_layers,
        "n_samples": len(norms),
    }


def compute_relative_thresholds(norm_stats):
    """Compute relative injection strength for each model."""
    results = {}
    for model_name, stats in norm_stats.items():
        alpha_max = MODEL_ALPHA_MAX.get(model_name, None)
        if alpha_max is None:
            continue
        mean_norm = stats["mean_norm"]
        ris = alpha_max / (mean_norm + 1e-12)
        stats_copy = dict(stats)
        stats_copy["alpha_max"] = alpha_max
        stats_copy["relative_injection_strength"] = ris
        stats_copy["model"] = model_name
        results[model_name] = stats_copy
    return results


def plot_results(results, output_path):
    """Visualise absolute vs relative steering thresholds."""
    models = list(results.keys())
    model_labels = [m.split("/")[-1][:15] for m in models]
    alpha_maxes = [results[m]["alpha_max"] for m in models]
    mean_norms = [results[m]["mean_norm"] for m in models]
    ris_values = [results[m]["relative_injection_strength"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # Left: Raw alpha_max
    ax = axes[0]
    bars = ax.bar(
        range(len(models)), alpha_maxes, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Max Raw |α| (before collapse)", fontsize=10)
    ax.set_title(
        "Raw Steering Bounds\n(confounded by residual scale)",
        fontsize=10,
        fontweight="bold",
    )
    for bar, val in zip(bars, alpha_maxes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Middle: Mean residual norm
    ax = axes[1]
    bars = ax.bar(
        range(len(models)), mean_norms, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Residual Stream Norm ||h||", fontsize=10)
    ax.set_title(
        f"Mean Residual Norm\n(at injection layer)", fontsize=10, fontweight="bold"
    )
    for bar, val in zip(bars, mean_norms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Relative injection strength
    ax = axes[2]
    bars = ax.bar(
        range(len(models)), ris_values, color=colors, edgecolor="black", linewidth=0.5
    )
    ris_mean = np.mean(ris_values)
    ris_std = np.std(ris_values)
    ax.axhline(
        ris_mean,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean RIS = {ris_mean:.3f}",
    )
    ax.fill_between(
        [-0.5, len(models) - 0.5],
        ris_mean - ris_std,
        ris_mean + ris_std,
        alpha=0.15,
        color="red",
        label=f"±1 SD = {ris_std:.3f}",
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative Injection Strength\n(α_max / mean ||h||)", fontsize=10)
    ax.set_title(
        "Normalised Steering Bound\n(RIS = α_max / mean ||h||)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    for bar, val in zip(bars, ris_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    cv = ris_std / (ris_mean + 1e-12) * 100  # coefficient of variation
    fig.suptitle(
        f"Residual Stream Norm Explains Cross-Model Steering Bound Variation\n"
        f"After normalisation: CV={cv:.1f}% (vs raw alpha CV={np.std(alpha_maxes) / np.mean(alpha_maxes) * 100:.0f}%)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Relative injection strength analysis")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODEL_ALPHA_MAX.keys()),
        help="List of models to analyze",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/relative_injection_results"
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Relative Injection Strength Analysis")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"{'=' * 60}\n")

    norm_stats = {}
    for model_name in args.models:
        print(f"\nProcessing: {model_name}")
        layer = MODEL_INJECTION_LAYERS.get(model_name, None)
        try:
            stats = get_residual_norm_at_layer(
                model_name, layer=layer, device=args.device
            )
            norm_stats[model_name] = stats
            print(f"    Mean norm: {stats['mean_norm']:.2f} ± {stats['std_norm']:.2f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Compute relative thresholds
    results = compute_relative_thresholds(norm_stats)

    print(f"\n{'=' * 60}")
    print(f"RELATIVE INJECTION STRENGTH SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<35} {'alpha_max':>10} {'mean_norm':>12} {'RIS':>10}")
    print(f"{'-' * 70}")
    ris_list = []
    for model_name, r in results.items():
        label = model_name.split("/")[-1][:30]
        print(
            f"{label:<35} {r['alpha_max']:>10.1f} {r['mean_norm']:>12.2f} {r['relative_injection_strength']:>10.4f}"
        )
        ris_list.append(r["relative_injection_strength"])

    if ris_list:
        cv_ris = np.std(ris_list) / (np.mean(ris_list) + 1e-12) * 100
        raw_alphas = [MODEL_ALPHA_MAX[m] for m in results]
        cv_raw = np.std(raw_alphas) / (np.mean(raw_alphas) + 1e-12) * 100
        print(f"\n  Raw alpha CV:    {cv_raw:.1f}%")
        print(f"  Normalised RIS CV: {cv_ris:.1f}%")
        if cv_ris < cv_raw * 0.6:
            print(
                "  ✓ Normalisation reduces variability by ≥40%: residual norm explains bounds."
            )
        else:
            print("  ⚠ Normalisation does not substantially reduce variability.")
    print(f"{'=' * 60}")

    # Save
    json_path = os.path.join(args.output_dir, "relative_injection_strengths.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    if len(results) >= 2:
        fig_path = os.path.join(args.output_dir, "relative_injection_strength.png")
        plot_results(results, fig_path)
        paper_fig = "paper/figures/relative_injection_strength.png"
        os.makedirs("paper/figures", exist_ok=True)
        plot_results(results, paper_fig)

    print("\n✓ Relative injection strength analysis complete!")


if __name__ == "__main__":
    main()
