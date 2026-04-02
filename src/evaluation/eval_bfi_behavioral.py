"""
Behavioral BFI Evaluation for Personality Steering

Replaces the flawed self-report approach (eval_bfi_self_report.py) with a behavioral
rubric evaluation:
  1. Steer model with persona vector at varying alpha values
  2. Present open-ended behavioral scenarios (NOT self-report questions)
  3. Generate free-form responses (~100 tokens, temperature=0.7)
  4. Score responses against BFI trait rubric indicators
  5. Repeat for confidence intervals

This avoids circular reasoning (steered model rating itself) and measures actual
behavioral manifestation of personality traits.

Usage:
    # Single model + trait
    python -m src.evaluation.eval_bfi_behavioral --model Qwen/Qwen3-0.6B --trait openness

    # All traits for a model
    python -m src.evaluation.eval_bfi_behavioral --model Qwen/Qwen3-0.6B --trait all

    # All models × all traits (sequential)
    python -m src.evaluation.eval_bfi_behavioral --all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.bfi_44_scale import (
    BEHAVIORAL_SCENARIOS,
    BFI_JUDGE_RUBRICS,
    score_behavioral_response,
)
from src.prompts.contrastive_prompts import apply_chat_template_safe

BIG_FIVE_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

ALL_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/gemma-2-2b-it",
]

DEFAULT_ALPHAS = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9
NUM_REPETITIONS = 3
DEFAULT_OUTPUT_DIR = "results/bfi_behavioral"


class SteeringEngine:
    """Manages activation steering via forward hooks on transformer layers."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = []
        self.steering_vec = None
        self.steering_layer = None
        self.alpha = 0.0
        self.active = False

    def _get_layers(self):
        """Locate transformer layers across different architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        raise ValueError("Cannot find transformer layers in model architecture")

    def _steering_hook(self, layer_idx):
        """Factory for per-layer forward hooks that inject steering vector."""

        def hook_fn(module, input, output):
            if (
                not self.active
                or layer_idx != self.steering_layer
                or self.steering_vec is None
            ):
                return output

            vec = self.alpha * self.steering_vec.to(self.device)
            if isinstance(output, tuple):
                modified = list(output)
                modified[0] = output[0] + vec.unsqueeze(0).unsqueeze(0)
                return tuple(modified)
            return output + vec.unsqueeze(0).unsqueeze(0)

        return hook_fn

    def setup(self, vec_path, layer_idx):
        """Load persona vector and register steering hooks."""
        vec = np.load(vec_path)
        self.steering_vec = torch.tensor(
            vec, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.steering_layer = layer_idx

        self.clear()
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._steering_hook(i))
            self.hooks.append(hook)

    def set_alpha(self, alpha):
        """Set steering strength and activate."""
        self.alpha = alpha
        self.active = True

    def deactivate(self):
        """Stop steering (alpha=0 equivalent, but avoids hook overhead)."""
        self.active = False

    def clear(self):
        """Remove all hooks and deactivate."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active = False


def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer with appropriate dtype."""
    print(f"Loading {model_name}...")
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
    return model, tokenizer


def generate_response(
    model, tokenizer, scenario, device, max_new_tokens=MAX_NEW_TOKENS
):
    """Generate a free-form response to a behavioral scenario."""
    messages = [
        {
            "role": "user",
            "content": (
                f"{scenario}\n\nPlease respond naturally as if you were in this situation. "
                "Describe your thoughts, feelings, and what you would do."
            ),
        }
    ]
    text = apply_chat_template_safe(
        tokenizer, messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def resolve_best_layer(model_short, trait):
    """Load best layer from analysis file, falling back to config mid-layer."""
    analysis_file = (
        f"results/persona_vectors/{model_short}/{trait}/analysis_v2_{trait}.json"
    )
    if os.path.exists(analysis_file):
        with open(analysis_file) as f:
            analysis = json.load(f)
            return analysis.get("best_layer_loso", analysis.get("best_layer_snr", 14))
    return None


def resolve_vector_path(model_short, trait, best_layer):
    """Construct path to persona vector .npy file."""
    return f"results/persona_vectors/{model_short}/{trait}/vectors/mean_diff_layer_{best_layer}.npy"


def evaluate_single_trait(model, tokenizer, steering, trait, alphas, device):
    """
    Run full behavioral evaluation for one trait.

    For each alpha × scenario × repetition:
      - Steer model
      - Generate behavioral response
      - Score against BFI rubric

    Returns dict of results.
    """
    scenarios = BEHAVIORAL_SCENARIOS[trait]
    results = {}

    for alpha in alphas:
        steering.set_alpha(alpha)

        alpha_data = {
            "alpha": alpha,
            "scenario_results": [],
        }

        for scenario_idx, scenario in enumerate(scenarios):
            for rep in range(NUM_REPETITIONS):
                response = generate_response(model, tokenizer, scenario, device)
                scoring = score_behavioral_response(response, trait)

                alpha_data["scenario_results"].append(
                    {
                        "scenario_idx": scenario_idx,
                        "scenario": scenario,
                        "repetition": rep,
                        "response": response,
                        "score": scoring["score"],
                        "high_matches": scoring["high_matches"],
                        "low_matches": scoring["low_matches"],
                    }
                )

        # Compute summary statistics for this alpha
        scores = [r["score"] for r in alpha_data["scenario_results"]]
        alpha_data["mean_score"] = float(np.mean(scores))
        alpha_data["std_score"] = float(np.std(scores))
        alpha_data["ci_lower"] = float(np.percentile(scores, 2.5))
        alpha_data["ci_upper"] = float(np.percentile(scores, 97.5))
        alpha_data["n_observations"] = len(scores)

        results[float(alpha)] = alpha_data
        print(
            f"    α={alpha:+5.1f}: mean={alpha_data['mean_score']:.2f} "
            f"± {alpha_data['std_score']:.2f} "
            f"[{alpha_data['ci_lower']:.2f}, {alpha_data['ci_upper']:.2f}]"
        )

    steering.deactivate()
    return results


def run_evaluation(model_name, trait, output_dir, device, alphas=None):
    """
    Load model, setup steering, and run behavioral evaluation.

    Args:
        model_name: HuggingFace model identifier
        trait: Big Five trait name or "all"
        output_dir: Directory to save results
        device: "cuda" or "cpu"
        alphas: List of steering strengths (default: DEFAULT_ALPHAS)
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    model_short = model_name.replace("/", "_")
    traits = BIG_FIVE_TRAITS if trait == "all" else [trait]

    # Validate traits have vectors
    valid_traits = []
    for t in traits:
        if t not in BEHAVIORAL_SCENARIOS:
            print(f"  WARNING: No behavioral scenarios for '{t}', skipping.")
            continue
        best_layer = resolve_best_layer(model_short, t)
        if best_layer is None:
            print(f"  WARNING: No analysis file for {model_short}/{t}, skipping.")
            continue
        vec_path = resolve_vector_path(model_short, t, best_layer)
        if not os.path.exists(vec_path):
            print(f"  WARNING: Vector not found: {vec_path}, skipping.")
            continue
        valid_traits.append((t, best_layer, vec_path))

    if not valid_traits:
        print("ERROR: No valid traits to evaluate. Check persona_vectors directory.")
        return False

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    steering = SteeringEngine(model, device)

    for t, best_layer, vec_path in valid_traits:
        print(f"\n{'=' * 60}")
        print(f"  Behavioral BFI Evaluation: {t}")
        print(f"  Model: {model_name}")
        print(f"  Steering Layer: L{best_layer}")
        print(
            f"  Scenarios: {len(BEHAVIORAL_SCENARIOS[t])} × {NUM_REPETITIONS} reps = "
            f"{len(BEHAVIORAL_SCENARIOS[t]) * NUM_REPETITIONS} observations per α"
        )
        print(f"  Alpha values: {alphas}")
        print(f"{'=' * 60}")

        steering.setup(vec_path, best_layer)

        t0 = time.time()
        trait_results = evaluate_single_trait(
            model, tokenizer, steering, t, alphas, device
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        dose_response = {
            "alphas": alphas,
            "means": [trait_results[float(a)]["mean_score"] for a in alphas],
            "stds": [trait_results[float(a)]["std_score"] for a in alphas],
            "ci_lowers": [trait_results[float(a)]["ci_lower"] for a in alphas],
            "ci_uppers": [trait_results[float(a)]["ci_upper"] for a in alphas],
        }

        trait_output_dir = os.path.join(output_dir, model_short)
        os.makedirs(trait_output_dir, exist_ok=True)

        output_data = {
            "model": model_name,
            "trait": t,
            "best_layer": int(best_layer),
            "alphas": [float(a) for a in alphas],
            "num_scenarios": len(BEHAVIORAL_SCENARIOS[t]),
            "num_repetitions": NUM_REPETITIONS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dose_response": dose_response,
            "results": {str(k): v for k, v in trait_results.items()},
            "elapsed_seconds": round(elapsed, 1),
        }

        output_file = os.path.join(trait_output_dir, f"bfi_behavioral_{t}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to: {output_file}")

    steering.clear()
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral BFI evaluation for personality steering"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (e.g., Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="openness",
        help="Trait to evaluate, or 'all' for all Big Five traits",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models × all Big Five traits (sequential)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (auto-detected if omitted)"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values (e.g., '-8,-4,0,4,8')",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"Free VRAM: {free_mem:.1f} GB")

    alphas = DEFAULT_ALPHAS
    if args.alphas:
        alphas = [float(a) for a in args.alphas.split(",")]

    if args.all:
        print(f"\n{'#' * 60}")
        print(f"  Running ALL models × ALL Big Five traits")
        print(f"  Models: {len(ALL_MODELS)}")
        print(f"  Traits: {BIG_FIVE_TRAITS}")
        print(f"  Total evaluations: {len(ALL_MODELS) * len(BIG_FIVE_TRAITS)}")
        print(f"{'#' * 60}")

        for i, model_name in enumerate(ALL_MODELS):
            print(f"\n[{i + 1}/{len(ALL_MODELS)}] {model_name}")
            success = run_evaluation(
                model_name=model_name,
                trait="all",
                output_dir=args.output_dir,
                device=device,
                alphas=alphas,
            )
            if not success:
                print(f"  FAILED: {model_name}")
            if device == "cuda" and i < len(ALL_MODELS) - 1:
                print("  Cooling GPU for 30s...")
                time.sleep(30)
        return

    if args.model is None:
        print("ERROR: --model is required (or use --all)")
        parser.print_help()
        return

    run_evaluation(
        model_name=args.model,
        trait=args.trait,
        output_dir=args.output_dir,
        device=device,
        alphas=alphas,
    )


if __name__ == "__main__":
    main()
