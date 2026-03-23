"""
Position-Swapped Control Experiment for Causal Patching

Addresses Reviewer Concern: "User token patching dominance might be a positional
artifact (recency effect) rather than genuine causal importance."

Design:
  Normal:  System=[Persona Instruction], User=[Neutral Scenario]
  Swapped: System=[Neutral Scenario],    User=[Persona Instruction]

If user-token dominance is purely positional (recency), then:
  - In BOTH conditions, user tokens (sequence-end) should dominate.
  
If user-token dominance reflects content-bound causal importance, then:
  - Normal:  user tokens dominate (neutral scenario at end)
  - Swapped: user tokens dominate (persona instruction at end)
  - The CAUSAL EFFECT should follow the persona instruction content,
    regardless of its position.

We measure: KL divergence from patching system-span vs user-span in both conditions.

Usage:
    PYTHONPATH=/home/fqwqf/persona python src/evaluation/eval_position_swap.py \
        --model Qwen/Qwen3-0.6B --trait openness --device cuda
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
from tqdm import tqdm
from src.prompts.contrastive_prompts import (
    apply_chat_template_safe,
    get_contrastive_pairs,
    BIG_FIVE_PROMPTS,
)


class PositionSwapPatcher:
    """Causal patcher that supports both normal and position-swapped prompt layouts."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        raise ValueError("Cannot find transformer layers")

    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _tokenize(self, messages):
        text = apply_chat_template_safe(
            self.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _get_logits(self, inputs):
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1, :].cpu().float()

    def _kl_div(self, clean_logits, patched_logits):
        p = torch.softmax(clean_logits, dim=-1)
        q = torch.softmax(patched_logits, dim=-1)
        return float(torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))))

    def _cache_all_activations(self, inputs):
        cache = {}
        layers = self._get_layers()

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    cache[layer_idx] = output[0].detach().clone()
                else:
                    cache[layer_idx] = output.detach().clone()

            return hook_fn

        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            self.model(**inputs)
        self._clear_hooks()
        return cache

    def _find_token_spans(self, messages):
        """Find system vs user token spans."""
        full_text = apply_chat_template_safe(
            self.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        full_tokens = self.tokenizer.encode(full_text)

        sys_only_text = apply_chat_template_safe(
            self.tokenizer, [messages[0]], tokenize=False, add_generation_prompt=False
        )
        sys_tokens = self.tokenizer.encode(sys_only_text)
        sys_end = len(sys_tokens)

        return {
            "system": (0, sys_end),
            "user": (sys_end, len(full_tokens)),
            "total": len(full_tokens),
        }

    def compute_patching_importance(self, pos_msgs, neg_msgs):
        """Compute KL divergence for system-span and user-span patching."""
        pos_inputs = self._tokenize(pos_msgs)
        neg_inputs = self._tokenize(neg_msgs)

        clean_logits = self._get_logits(pos_inputs)
        neg_cache = self._cache_all_activations(neg_inputs)

        spans = self._find_token_spans(pos_msgs)
        n_layers = len(self._get_layers())
        layers = self._get_layers()

        results = {
            "system_tokens": np.zeros(n_layers),
            "user_tokens": np.zeros(n_layers),
        }

        for target_layer in range(n_layers):
            if target_layer not in neg_cache:
                continue

            for span_name in ["system", "user"]:
                start, end = spans[span_name]

                def make_span_patch(neg_h, s, e):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            mod = list(output)
                            patched = output[0].clone()
                            min_t = min(e, patched.shape[1], neg_h.shape[1])
                            s_clamped = min(s, min_t)
                            patched[:, s_clamped:min_t, :] = neg_h[
                                :, s_clamped:min_t, :
                            ]
                            mod[0] = patched
                            return tuple(mod)
                        else:
                            patched = output.clone()
                            min_t = min(e, patched.shape[1], neg_h.shape[1])
                            s_clamped = min(s, min_t)
                            patched[:, s_clamped:min_t, :] = neg_h[
                                :, s_clamped:min_t, :
                            ]
                            return patched

                    return hook_fn

                hook = layers[target_layer].register_forward_hook(
                    make_span_patch(neg_cache[target_layer], start, end)
                )
                self.hooks = [hook]
                patched_logits = self._get_logits(pos_inputs)
                self._clear_hooks()
                results[f"{span_name}_tokens"][target_layer] = self._kl_div(
                    clean_logits, patched_logits
                )

        return results, spans


def create_swapped_pairs(trait_name, n_pairs=10):
    """
    Create position-swapped contrastive pairs.

    Normal:  System=[High/Low persona], User=[Scenario]
    Swapped: System=[Scenario],         User=[High/Low persona]
    """
    import random

    rng = random.Random(42 + hash(trait_name))

    data = BIG_FIVE_PROMPTS[trait_name]
    scenarios = data["scenarios"][:n_pairs]

    normal_pairs = []
    swapped_pairs = []

    for scenario in scenarios:
        high_sys = rng.choice(data["high_system"])
        low_sys = rng.choice(data["low_system"])

        # Normal layout: persona in system, scenario in user
        normal_pos = [
            {"role": "system", "content": high_sys},
            {"role": "user", "content": scenario},
        ]
        normal_neg = [
            {"role": "system", "content": low_sys},
            {"role": "user", "content": scenario},
        ]

        # Swapped layout: scenario in system, persona in user
        swapped_pos = [
            {"role": "system", "content": scenario},
            {"role": "user", "content": high_sys},
        ]
        swapped_neg = [
            {"role": "system", "content": scenario},
            {"role": "user", "content": low_sys},
        ]

        normal_pairs.append((normal_pos, normal_neg))
        swapped_pairs.append((swapped_pos, swapped_neg))

    return normal_pairs, swapped_pairs


def run_position_swap_experiment(model, tokenizer, trait_name, device, n_pairs=10):
    """Run full position-swap control experiment."""
    patcher = PositionSwapPatcher(model, tokenizer, device)
    normal_pairs, swapped_pairs = create_swapped_pairs(trait_name, n_pairs)

    n_layers = model.config.num_hidden_layers
    conditions = {
        "normal": {"system_tokens": [], "user_tokens": []},
        "swapped": {"system_tokens": [], "user_tokens": []},
    }

    # Run normal condition
    print(f"  Running NORMAL condition ({len(normal_pairs)} pairs)...")
    for pos_msgs, neg_msgs in tqdm(normal_pairs, desc="Normal"):
        results, spans = patcher.compute_patching_importance(pos_msgs, neg_msgs)
        conditions["normal"]["system_tokens"].append(results["system_tokens"])
        conditions["normal"]["user_tokens"].append(results["user_tokens"])

    # Run swapped condition
    print(f"  Running SWAPPED condition ({len(swapped_pairs)} pairs)...")
    for pos_msgs, neg_msgs in tqdm(swapped_pairs, desc="Swapped"):
        results, spans = patcher.compute_patching_importance(pos_msgs, neg_msgs)
        conditions["swapped"]["system_tokens"].append(results["system_tokens"])
        conditions["swapped"]["user_tokens"].append(results["user_tokens"])

    # Aggregate
    summary = {}
    for cond in ["normal", "swapped"]:
        summary[cond] = {}
        for span in ["system_tokens", "user_tokens"]:
            arr = np.array(conditions[cond][span])
            summary[cond][span] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "per_sample": arr.tolist(),
            }

    # Compute key metrics for interpretation
    for cond in ["normal", "swapped"]:
        sys_total = np.sum(summary[cond]["system_tokens"]["mean"])
        usr_total = np.sum(summary[cond]["user_tokens"]["mean"])
        summary[cond]["total_system_kl"] = float(sys_total)
        summary[cond]["total_user_kl"] = float(usr_total)
        summary[cond]["user_dominance_ratio"] = float(usr_total / (sys_total + 1e-10))

    # Interpretation
    normal_ratio = summary["normal"]["user_dominance_ratio"]
    swapped_ratio = summary["swapped"]["user_dominance_ratio"]

    summary["interpretation"] = {
        "normal_user_dominance_ratio": normal_ratio,
        "swapped_user_dominance_ratio": swapped_ratio,
        "position_effect": (
            "POSITIONAL_CONFOUND" if swapped_ratio > 2.0 else "CONTENT_BOUND"
        ),
        "explanation": (
            "If swapped ratio >> 1: user tokens still dominate even when they "
            "contain persona instruction (positional confound). "
            "If swapped ratio ~ 1 or < 1: causal effect follows persona content "
            "regardless of position (content-bound, no positional confound)."
        ),
    }

    return summary


def plot_position_swap_results(summary, trait_name, output_path):
    """Create 2x2 comparison figure with unified Y-axes for fair comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_layers = len(summary["normal"]["system_tokens"]["mean"])
    layers = list(range(n_layers))

    conditions = [
        ("normal", "Normal Layout\n(System=Persona, User=Scenario)"),
        ("swapped", "Swapped Layout\n(System=Scenario, User=Persona)"),
    ]

    # Pre-compute unified Y limits for fair side-by-side comparison
    all_line_vals = []
    all_bar_vals = []
    for cond, _ in conditions:
        sys_mean = np.array(summary[cond]["system_tokens"]["mean"])
        usr_mean = np.array(summary[cond]["user_tokens"]["mean"])
        sys_std = np.array(summary[cond]["system_tokens"]["std"])
        usr_std = np.array(summary[cond]["user_tokens"]["std"])
        all_line_vals.extend((usr_mean + usr_std).tolist())
        all_line_vals.extend((sys_mean + sys_std).tolist())
        all_bar_vals.append(summary[cond]["total_system_kl"])
        all_bar_vals.append(summary[cond]["total_user_kl"])

    line_ymax = max(all_line_vals) * 1.15 if all_line_vals else 1.0
    bar_ymax = max(all_bar_vals) * 1.15 if all_bar_vals else 1.0

    for col, (cond, title) in enumerate(conditions):
        # Top row: line plots (unified Y)
        ax = axes[0, col]
        sys_mean = np.array(summary[cond]["system_tokens"]["mean"])
        usr_mean = np.array(summary[cond]["user_tokens"]["mean"])
        sys_std = np.array(summary[cond]["system_tokens"]["std"])
        usr_std = np.array(summary[cond]["user_tokens"]["std"])

        ax.plot(
            layers,
            sys_mean,
            "o-",
            color="#FF5722",
            linewidth=2,
            label="System tokens",
            markersize=4,
        )
        ax.fill_between(
            layers, sys_mean - sys_std, sys_mean + sys_std, alpha=0.15, color="#FF5722"
        )
        ax.plot(
            layers,
            usr_mean,
            "s-",
            color="#4CAF50",
            linewidth=2,
            label="User tokens",
            markersize=4,
        )
        ax.fill_between(
            layers, usr_mean - usr_std, usr_mean + usr_std, alpha=0.15, color="#4CAF50"
        )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("KL Divergence")
        ax.set_ylim(0, line_ymax)  # <-- unified Y-axis
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom row: total KL bar charts (unified Y)
        ax2 = axes[1, col]
        sys_total = summary[cond]["total_system_kl"]
        usr_total = summary[cond]["total_user_kl"]

        bars = ax2.bar(
            ["System\ntokens", "User\ntokens"],
            [sys_total, usr_total],
            color=["#FF5722", "#4CAF50"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_ylabel("Total KL Divergence (sum over layers)")
        ax2.set_title(f"Total Causal Effect ({cond.capitalize()})", fontsize=10)
        ax2.set_ylim(0, bar_ymax)  # <-- unified Y-axis

        # Annotate bars
        for bar, val in zip(bars, [sys_total, usr_total]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar_ymax * 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle(
        f"Position-Swapped Control Experiment — {trait_name.capitalize()}\n"
        f"Testing whether causal patching results reflect content or position",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Position-swapped control for causal patching"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trait", type=str, default="openness")
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument(
        "--output_dir", type=str, default="results/position_swap_results"
    )
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
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Position-Swapped Control Experiment")
    print(f"Model: {args.model} | Trait: {args.trait} | Pairs: {args.n_pairs}")
    print(f"{'=' * 60}\n")

    summary = run_position_swap_experiment(
        model, tokenizer, args.trait, args.device, args.n_pairs
    )

    # Save results
    json_path = os.path.join(output_dir, f"position_swap_{args.trait}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Print interpretation
    interp = summary["interpretation"]
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  Normal:  user dominance ratio = {interp['normal_user_dominance_ratio']:.2f}"
    )
    print(
        f"  Swapped: user dominance ratio = {interp['swapped_user_dominance_ratio']:.2f}"
    )
    print(f"  Verdict: {interp['position_effect']}")
    print(f"{'=' * 60}")

    # Plot
    fig_path = os.path.join(output_dir, f"position_swap_{args.trait}.png")
    plot_position_swap_results(summary, args.trait, fig_path)

    # Also save to paper figures
    paper_fig_path = "paper/figures/position_swap_control.png"
    os.makedirs("paper/figures", exist_ok=True)
    plot_position_swap_results(summary, args.trait, paper_fig_path)

    print("\n✓ Position-swap control experiment complete!")


if __name__ == "__main__":
    main()
