"""
LLM-as-Judge Evaluation for Personality Steering

This script uses GPT-4 (or similar) as a judge to rate generated text on BFI traits.
This provides expert-level psychological assessment of steering effectiveness.

Usage:
    python eval_llm_as_judge.py --model Qwen/Qwen3-0.6B --trait openness --judge gpt-4

Requires:
    - OpenAI API key in environment: export OPENAI_API_KEY=your_key
    - Or use local judge model with --judge-local flag
"""

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time

from src.evaluation.bfi_44_scale import (
    BFI_JUDGE_RUBRICS,
    get_judge_prompt,
    parse_judge_response,
)
from src.prompts.contrastive_prompts import apply_chat_template_safe


def call_openai_judge(prompt, model="gpt-4", max_retries=3):
    """Call OpenAI API for judging."""
    try:
        import openai
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert psychologist evaluating personality traits.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Deterministic for consistency
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
                continue
            else:
                raise e


class LLMJudgeEvaluator:
    """Evaluates steering using LLM-as-Judge."""

    def __init__(self, model, tokenizer, device, judge_model="gpt-4"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.judge_model = judge_model
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

    def generate(self, prompt, max_new_tokens=150, temperature=0.7):
        """Generate text with current steering."""
        messages = [{"role": "user", "content": prompt}]
        text = apply_chat_template_safe(
            self.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    def judge_text(self, text, trait_name):
        """
        Use LLM judge to rate text on trait.

        Returns:
            Tuple of (rating, reasoning)
        """
        judge_prompt = get_judge_prompt(text, trait_name)

        if self.judge_model.startswith("gpt"):
            response = call_openai_judge(judge_prompt, model=self.judge_model)
        else:
            # Could add support for local judge models here
            raise NotImplementedError(
                f"Judge model {self.judge_model} not supported yet"
            )

        rating, reasoning = parse_judge_response(response)
        return rating, reasoning

    def evaluate_alpha_sweep(self, trait_name, alphas, eval_prompts):
        """
        Evaluate trait ratings across different steering strengths.

        Args:
            trait_name: Trait to evaluate
            alphas: List of alpha values to test
            eval_prompts: List of prompts to generate responses for

        Returns:
            Dict with results for each alpha
        """
        results = {}

        for alpha in alphas:
            self.alpha = alpha
            self.active = True

            prompt_results = []
            ratings = []

            print(f"\n  α={alpha:+6.1f}:")
            for prompt in tqdm(
                eval_prompts, desc=f"    Generating & judging", leave=False
            ):
                # Generate response
                response = self.generate(prompt)

                # Judge response
                rating, reasoning = self.judge_text(response, trait_name)

                if rating is not None:
                    ratings.append(rating)

                prompt_results.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "rating": rating,
                        "reasoning": reasoning,
                    }
                )

                # Rate limiting for API
                if self.judge_model.startswith("gpt"):
                    time.sleep(0.5)

            results[float(alpha)] = {
                "mean_rating": float(np.mean(ratings)) if ratings else None,
                "std_rating": float(np.std(ratings)) if ratings else None,
                "prompt_results": prompt_results,
                "n_rated": len(ratings),
            }

            if ratings:
                print(
                    f"    Mean Rating: {np.mean(ratings):.2f} ± {np.std(ratings):.2f}"
                )
            else:
                print(f"    Mean Rating: N/A (no valid ratings)")

        self.active = False
        return results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model to evaluate"
    )
    parser.add_argument("--trait", type=str, required=True, help="Trait to evaluate")
    parser.add_argument(
        "--judge",
        type=str,
        default="gpt-4",
        help="Judge model (gpt-4, gpt-4-turbo, etc.)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="llm_judge_results", help="Output directory"
    )
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument(
        "--n_prompts", type=int, default=5, help="Number of evaluation prompts"
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # Check if trait has judge rubric
    if args.trait not in BFI_JUDGE_RUBRICS:
        print(f"ERROR: Trait '{args.trait}' not in BFI judge rubrics.")
        print(f"Available traits: {', '.join(BFI_JUDGE_RUBRICS.keys())}")
        return

    # Check API key for OpenAI models
    if args.judge.startswith("gpt") and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it: export OPENAI_API_KEY=your_key")
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

    # Evaluation prompts
    eval_prompts = [
        "Tell me about your approach to learning new things.",
        "How do you handle unexpected challenges?",
        "Describe your ideal weekend.",
        "What's your perspective on taking risks?",
        "How do you make important decisions?",
    ][: args.n_prompts]

    print(f"\n{'=' * 60}")
    print(f"LLM-as-Judge Evaluation: {args.trait}")
    print(f"Model: {args.model}")
    print(f"Judge: {args.judge}")
    print(f"Best Layer: L{best_layer}")
    print(f"Evaluation Prompts: {len(eval_prompts)}")
    print(f"{'=' * 60}")

    # Setup evaluator
    evaluator = LLMJudgeEvaluator(model, tokenizer, args.device, judge_model=args.judge)
    evaluator.setup_steering(vec_path, best_layer)

    # Alpha sweep
    alphas = [-6.0, -3.0, 0.0, 3.0, 6.0]

    print("\nRunning LLM-as-Judge evaluation across α values...")
    print("(This may take several minutes due to API rate limits)")
    results = evaluator.evaluate_alpha_sweep(args.trait, alphas, eval_prompts)

    evaluator._clear_hooks()

    # Save results
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "model": args.model,
        "trait": args.trait,
        "judge_model": args.judge,
        "best_layer": int(best_layer),
        "alphas": alphas,
        "eval_prompts": eval_prompts,
        "results": results,
    }

    output_file = os.path.join(output_dir, f"llm_judge_{args.trait}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Summary
    print("\nSummary:")
    for alpha in alphas:
        mean_rating = results[alpha]["mean_rating"]
        if mean_rating is not None:
            print(f"  α={alpha:+6.1f}: Mean Rating = {mean_rating:.2f}")

    print("\nInterpretation:")
    print("- Ratings range from 1 (very low trait) to 5 (very high trait)")
    print("- Positive α should increase ratings, negative α should decrease")
    print("- This provides expert-level psychological assessment of steering")


if __name__ == "__main__":
    main()
