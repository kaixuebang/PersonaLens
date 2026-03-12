"""
BFI-44 Self-Report Evaluation for Steering

This script evaluates personality steering by having the model complete the BFI-44
questionnaire under different steering conditions (α values).

The model answers BFI-44 items directly, and we compute trait scores from its responses.
This provides a validated psychological assessment of steering effectiveness.

Usage:
    python eval_bfi_self_report.py --model Qwen/Qwen3-0.6B --trait openness
"""

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.evaluation.bfi_44_scale import (
    BFI_44_ITEMS,
    get_bfi_prompt,
    compute_bfi_score_from_ratings,
)
from src.prompts.contrastive_prompts import apply_chat_template_safe


class BFISelfReportEvaluator:
    """Evaluates steering using BFI-44 self-report."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
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
        """Hook that adds steering vector to hidden state."""

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
            else:
                return output + vec.unsqueeze(0).unsqueeze(0)

        return hook_fn

    def setup_steering(self, vec_path, layer_idx):
        """Setup steering with persona vector."""
        vec = np.load(vec_path)
        self.steering_vec = torch.tensor(
            vec, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.steering_layer = layer_idx

        self._clear_hooks()
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._steering_hook(i))
            self.hooks.append(hook)

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active = False

    def get_bfi_rating(self, item_text):
        """
        Get model's rating for a BFI item.
        Returns rating (1-5) based on logit probabilities.
        """
        prompt = get_bfi_prompt(item_text, response_format="likert")
        messages = [{"role": "user", "content": prompt}]
        text = apply_chat_template_safe(
            self.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]

        # Get token IDs for digits 1-5
        token_ids = []
        for i in range(1, 6):
            # Try different encodings
            candidates = [
                self.tokenizer.encode(str(i), add_special_tokens=False),
                self.tokenizer.encode(f" {i}", add_special_tokens=False),
                self.tokenizer.encode(f"{i}", add_special_tokens=False),
            ]
            # Use the last token from the first non-empty encoding
            for cand in candidates:
                if cand:
                    token_ids.append(cand[-1])
                    break

        if len(token_ids) != 5:
            # Fallback: use most common single-digit token pattern
            token_ids = [self.tokenizer.encode(str(i))[-1] for i in range(1, 6)]

        # Extract logits and compute expected rating
        rating_logits = next_token_logits[token_ids]
        probs = torch.softmax(rating_logits, dim=-1)

        # Expected value (1*p1 + 2*p2 + 3*p3 + 4*p4 + 5*p5)
        expected_rating = sum((i + 1) * probs[i].item() for i in range(5))

        return expected_rating

    def evaluate_alpha_sweep(self, trait_name, alphas):
        """
        Evaluate BFI scores across different steering strengths.

        Args:
            trait_name: Trait to evaluate
            alphas: List of alpha values to test

        Returns:
            Dict with results for each alpha
        """
        items = BFI_44_ITEMS[trait_name]
        all_items = items["positive"] + items["negative"]

        results = {}

        for alpha in alphas:
            self.alpha = alpha
            self.active = True

            ratings = {}
            item_responses = []

            print(f"\n  α={alpha:+6.1f}:")
            for item in tqdm(all_items, desc=f"    BFI items", leave=False):
                rating = self.get_bfi_rating(item)
                ratings[item] = rating
                item_responses.append({"item": item, "rating": float(rating)})

            # Compute trait score
            trait_score = compute_bfi_score_from_ratings(ratings, trait_name)

            results[float(alpha)] = {
                "trait_score": float(trait_score) if trait_score is not None else None,
                "item_responses": item_responses,
                "mean_rating": float(np.mean([r["rating"] for r in item_responses])),
                "std_rating": float(np.std([r["rating"] for r in item_responses])),
            }

            print(
                f"    Trait Score: {trait_score:.2f}"
                if trait_score
                else "    Trait Score: N/A"
            )

        self.active = False
        return results


def main():
    parser = argparse.ArgumentParser(description="BFI-44 self-report evaluation")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name"
    )
    parser.add_argument("--trait", type=str, required=True, help="Trait to evaluate")
    parser.add_argument(
        "--output_dir", type=str, default="bfi_results", help="Output directory"
    )
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # Check if trait has BFI items
    if args.trait not in BFI_44_ITEMS:
        print(f"ERROR: Trait '{args.trait}' not in BFI-44 items.")
        print(f"Available traits: {', '.join(BFI_44_ITEMS.keys())}")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
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

    # Load persona vector
    model_short = args.model.replace("/", "_")
    vectors_dir = f"persona_vectors/{model_short}/{args.trait}/vectors"

    # Find best layer
    analysis_file = (
        f"persona_vectors/{model_short}/{args.trait}/analysis_v2_{args.trait}.json"
    )
    if os.path.exists(analysis_file):
        with open(analysis_file) as f:
            analysis = json.load(f)
            best_layer = analysis.get(
                "best_layer_loso", analysis.get("best_layer_snr", 14)
            )
    else:
        best_layer = model.config.num_hidden_layers // 2

    vec_path = os.path.join(vectors_dir, f"mean_diff_layer_{best_layer}.npy")
    if not os.path.exists(vec_path):
        print(f"ERROR: Persona vector not found: {vec_path}")
        print("Please run extraction first.")
        return

    print(f"\n{'=' * 60}")
    print(f"BFI-44 Self-Report Evaluation: {args.trait}")
    print(f"Model: {args.model}")
    print(f"Best Layer: L{best_layer}")
    print(f"{'=' * 60}")

    # Setup evaluator
    evaluator = BFISelfReportEvaluator(model, tokenizer, args.device)
    evaluator.setup_steering(vec_path, best_layer)

    # Alpha sweep
    alphas = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

    print("\nRunning BFI-44 evaluation across α values...")
    results = evaluator.evaluate_alpha_sweep(args.trait, alphas)

    evaluator._clear_hooks()

    # Save results
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "model": args.model,
        "trait": args.trait,
        "best_layer": int(best_layer),
        "alphas": alphas,
        "results": results,
    }

    output_file = os.path.join(output_dir, f"bfi_self_report_{args.trait}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Summary
    print("\nSummary:")
    for alpha in alphas:
        score = results[alpha]["trait_score"]
        if score is not None:
            print(f"  α={alpha:+6.1f}: BFI Score = {score:.2f}")

    print("\nInterpretation:")
    print("- BFI scores range from 1 (low trait) to 5 (high trait)")
    print("- Positive α should increase scores, negative α should decrease")
    print("- This validates steering using a standardized psychological scale")


if __name__ == "__main__":
    main()
