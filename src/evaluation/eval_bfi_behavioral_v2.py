"""
Behavioral BFI Evaluation v2 — Response Generation Phase

Phase 1 of the LLM-as-Judge evaluation pipeline:
  1. Steer model with persona vector at varying alpha values
  2. Present targeted V2 behavioral scenarios (forced-choice style)
  3. Generate free-form responses (~200 tokens, temperature=0.8)
  4. Save raw responses for Phase 2 (judge scoring)

Key improvements over v1:
  - V2 scenarios: 7-8 targeted forced-choice scenarios per trait
  - 5 repetitions per (alpha, scenario) instead of 3
  - 7 alpha values instead of 9 (faster, covers the key range)
  - Saves raw responses without keyword scoring (LLM-as-Judge in Phase 2)

Usage:
    python -m src.evaluation.eval_bfi_behavioral_v2 --model Qwen/Qwen3-0.6B --trait openness
    python -m src.evaluation.eval_bfi_behavioral_v2 --all
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.bfi_44_scale import BEHAVIORAL_SCENARIOS_V2
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

DEFAULT_ALPHAS = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_P = 0.9
NUM_REPETITIONS = 5
DEFAULT_OUTPUT_DIR = "results/bfi_behavioral_v2"


class SteeringEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = []
        self.steering_vec = None
        self.steering_layer = None
        self.alpha = 0.0
        self.active = False

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        raise ValueError("Cannot find transformer layers")

    def _steering_hook(self, layer_idx):
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
        self.alpha = alpha
        self.active = True

    def deactivate(self):
        self.active = False

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active = False


def generate_response(
    model, tokenizer, scenario, device, max_new_tokens=MAX_NEW_TOKENS
):
    messages = [
        {
            "role": "user",
            "content": (
                f"{scenario}\n\n"
                "Answer honestly about what you would actually do in this situation. "
                "Be specific about your thoughts, feelings, and the concrete actions you would take."
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def resolve_best_layer(model_short, trait):
    analysis_file = (
        f"results/persona_vectors/{model_short}/{trait}/analysis_v2_{trait}.json"
    )
    if os.path.exists(analysis_file):
        with open(analysis_file) as f:
            analysis = json.load(f)
            return analysis.get("best_layer_loso", analysis.get("best_layer_snr", 14))
    return None


def resolve_vector_path(model_short, trait, best_layer):
    return f"results/persona_vectors/{model_short}/{trait}/vectors/mean_diff_layer_{best_layer}.npy"


def evaluate_single_trait(model, tokenizer, steering, trait, alphas, device):
    scenarios = BEHAVIORAL_SCENARIOS_V2[trait]
    results = {}

    for alpha in alphas:
        steering.set_alpha(alpha)
        alpha_data = {"alpha": alpha, "scenario_results": []}

        for scenario_idx, scenario in enumerate(scenarios):
            for rep in range(NUM_REPETITIONS):
                response = generate_response(model, tokenizer, scenario, device)
                alpha_data["scenario_results"].append(
                    {
                        "scenario_idx": scenario_idx,
                        "scenario": scenario,
                        "repetition": rep,
                        "response": response,
                    }
                )

        results[float(alpha)] = alpha_data
        n = len(alpha_data["scenario_results"])
        print(f"    α={alpha:+5.1f}: {n} responses generated")

    steering.deactivate()
    return results


def run_evaluation(model_name, trait, output_dir, device, alphas=None,
                   layer_override=None):
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    model_short = model_name.replace("/", "_")
    traits = BIG_FIVE_TRAITS if trait == "all" else [trait]

    valid_traits = []
    for t in traits:
        if t not in BEHAVIORAL_SCENARIOS_V2:
            print(f"  WARNING: No V2 scenarios for '{t}', skipping.")
            continue
        if layer_override is not None:
            best_layer = layer_override if isinstance(layer_override, int) else layer_override[0]
            print(f"  Layer override: L{best_layer}")
        else:
            best_layer = resolve_best_layer(model_short, t)
        if best_layer is None:
            print(f"  WARNING: No analysis for {model_short}/{t}, skipping.")
            continue
        vec_path = resolve_vector_path(model_short, t, best_layer)
        if not os.path.exists(vec_path):
            print(f"  WARNING: Vector not found: {vec_path}, skipping.")
            continue
        valid_traits.append((t, best_layer, vec_path))

    if not valid_traits:
        print("ERROR: No valid traits to evaluate.")
        return False

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    steering = SteeringEngine(model, device)

    for t, best_layer, vec_path in valid_traits:
        n_scenarios = len(BEHAVIORAL_SCENARIOS_V2[t])
        print(f"\n{'=' * 60}")
        print(f"  V2 Behavioral Eval: {t}")
        print(f"  Model: {model_name}")
        print(f"  Layer: L{best_layer}")
        print(
            f"  {n_scenarios} scenarios × {NUM_REPETITIONS} reps × {len(alphas)} alphas = "
            f"{n_scenarios * NUM_REPETITIONS * len(alphas)} total responses"
        )
        print(f"{'=' * 60}")

        steering.setup(vec_path, best_layer)
        t0 = time.time()
        trait_results = evaluate_single_trait(
            model, tokenizer, steering, t, alphas, device
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        trait_output_dir = os.path.join(output_dir, model_short)
        os.makedirs(trait_output_dir, exist_ok=True)

        output_data = {
            "model": model_name,
            "trait": t,
            "best_layer": int(best_layer),
            "alphas": [float(a) for a in alphas],
            "num_scenarios": n_scenarios,
            "num_repetitions": NUM_REPETITIONS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "scoring_method": "llm_as_judge_pending",
            "results": {str(k): v for k, v in trait_results.items()},
            "elapsed_seconds": round(elapsed, 1),
        }

        output_file = os.path.join(trait_output_dir, f"responses_{t}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved to: {output_file}")

    steering.clear()
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral BFI v2 — response generation"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trait", type=str, default="openness")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alphas", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to override auto-selection")
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

    layer_override = None
    if args.layers:
        layer_override = [int(l) for l in args.layers.split(",")]
        if len(layer_override) == 1:
            layer_override = layer_override[0]

    if args.all:
        for i, model_name in enumerate(ALL_MODELS):
            print(f"\n[{i + 1}/{len(ALL_MODELS)}] {model_name}")
            run_evaluation(model_name, "all", args.output_dir, device, alphas,
                         layer_override=layer_override)
            if device == "cuda" and i < len(ALL_MODELS) - 1:
                print("  Cooling GPU 30s...")
                time.sleep(30)
        return

    if args.model is None:
        print("ERROR: --model required (or use --all)")
        return

    run_evaluation(args.model, args.trait, args.output_dir, device, alphas,
                  layer_override=layer_override)


if __name__ == "__main__":
    main()
