"""
Baseline Comparison Experiments

This script implements baseline comparisons to demonstrate that activation steering
outperforms simpler alternatives:

1. Prompt-Only Baseline: Pure prompt engineering without activation steering
2. Random Direction Baseline: Injecting random vectors instead of extracted persona vectors
3. Zero Baseline: No intervention (neutral model behavior)

Usage:
    python eval_baselines.py --model Qwen/Qwen3-0.6B --trait openness --alpha 3.0
"""

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.prompts.contrastive_prompts import apply_chat_template_safe


class BaselineEvaluator:
    """Evaluates different baseline approaches for personality steering."""

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

    def setup_steering(self, vec, layer_idx):
        """Setup steering with given vector."""
        self.steering_vec = vec
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

    def generate(self, prompt, max_new_tokens=512, temperature=0.7, do_sample=True):
        """Generate text with current steering configuration."""
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
                do_sample=do_sample,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    def evaluate_prompt_only(self, trait_name, trait_direction, eval_prompts):
        """
        Baseline 1: Prompt-only control.
        Add personality instruction to system prompt instead of activation steering.
        """
        results = []

        # Create personality-instructed prompt
        if trait_direction == "high":
            personality_instruction = self._get_high_trait_instruction(trait_name)
        else:
            personality_instruction = self._get_low_trait_instruction(trait_name)

        for prompt in tqdm(eval_prompts, desc="Prompt-Only"):
            # Add personality instruction to prompt
            instructed_prompt = f"{personality_instruction}\n\n{prompt}"
            response = self.generate(instructed_prompt)
            results.append(
                {"prompt": prompt, "response": response, "method": "prompt_only"}
            )

        return results

    def evaluate_random_direction(self, hidden_dim, layer_idx, alpha, eval_prompts):
        """
        Baseline 2: Random direction injection.
        Inject random vector instead of extracted persona vector.
        """
        # Generate random vector with same norm as typical persona vectors
        random_vec = np.random.randn(hidden_dim).astype(np.float32)
        random_vec = random_vec / np.linalg.norm(random_vec)
        random_vec = random_vec * 5.0  # Scale to typical persona vector magnitude

        random_vec_tensor = torch.tensor(
            random_vec, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self.setup_steering(random_vec_tensor, layer_idx)
        self.alpha = alpha
        self.active = True

        results = []
        for prompt in tqdm(eval_prompts, desc="Random Direction"):
            response = self.generate(prompt)
            results.append(
                {"prompt": prompt, "response": response, "method": "random_direction"}
            )

        self._clear_hooks()
        return results

    def evaluate_zero_baseline(self, eval_prompts):
        """
        Baseline 3: Zero intervention (neutral model).
        No steering, no personality instruction.
        """
        self._clear_hooks()

        results = []
        for prompt in tqdm(eval_prompts, desc="Zero Baseline"):
            response = self.generate(prompt)
            results.append(
                {"prompt": prompt, "response": response, "method": "zero_baseline"}
            )

        return results

    def evaluate_activation_steering(self, persona_vec, layer_idx, alpha, eval_prompts):
        """
        Main method: Activation steering with extracted persona vector.
        """
        persona_vec_tensor = torch.tensor(
            persona_vec, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self.setup_steering(persona_vec_tensor, layer_idx)
        self.alpha = alpha
        self.active = True

        results = []
        for prompt in tqdm(eval_prompts, desc="Activation Steering"):
            response = self.generate(prompt)
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "method": "activation_steering",
                }
            )

        self._clear_hooks()
        return results

    def _get_high_trait_instruction(self, trait_name):
        """Get personality instruction for high trait level."""
        instructions = {
            "openness": "You are extremely open to new experiences, intellectually curious, and imaginative.",
            "conscientiousness": "You are highly organized, disciplined, and goal-oriented.",
            "extraversion": "You are very outgoing, energetic, and sociable.",
            "agreeableness": "You are extremely warm, empathetic, and cooperative.",
            "neuroticism": "You experience emotions intensely and worry frequently.",
        }
        return instructions.get(trait_name, f"You have high {trait_name}.")

    def _get_low_trait_instruction(self, trait_name):
        """Get personality instruction for low trait level."""
        instructions = {
            "openness": "You are very practical, conventional, and prefer routine.",
            "conscientiousness": "You are spontaneous, flexible, and go with the flow.",
            "extraversion": "You are quiet, reserved, and introspective.",
            "agreeableness": "You are blunt, competitive, and skeptical.",
            "neuroticism": "You are emotionally stable, calm, and resilient.",
        }
        return instructions.get(trait_name, f"You have low {trait_name}.")


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison experiments")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name"
    )
    parser.add_argument("--trait", type=str, required=True, help="Trait name")
    parser.add_argument("--alpha", type=float, default=3.0, help="Steering strength")
    parser.add_argument(
        "--direction",
        type=str,
        default="high",
        choices=["high", "low"],
        help="Trait direction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baseline_results",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

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
    vectors_dir = f"results/persona_vectors/{model_short}/{args.trait}/vectors"

    # Find best layer from analysis
    analysis_file = f"results/persona_vectors/{model_short}/{args.trait}/analysis_v2_{args.trait}.json"
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
        print(
            "Please run extraction first: python src/extraction/extract_persona_vectors_v2.py"
        )
        return

    persona_vec = np.load(vec_path)
    hidden_dim = persona_vec.shape[0]

    # Evaluation prompts
    eval_prompts = [
        "Tell me about your approach to learning new things.",
        "How do you handle unexpected challenges?",
        "Describe your ideal weekend.",
        "What's your perspective on taking risks?",
        "How do you make important decisions?",
    ]

    print(f"\n{'=' * 60}")
    print(f"Baseline Comparison: {args.trait} ({args.direction})")
    print(f"Model: {args.model}")
    print(f"Best Layer: L{best_layer}")
    print(f"Alpha: {args.alpha}")
    print(f"{'=' * 60}\n")

    evaluator = BaselineEvaluator(model, tokenizer, args.device)

    # Run all baselines
    print("[1/4] Zero Baseline (no intervention)...")
    zero_results = evaluator.evaluate_zero_baseline(eval_prompts)

    print("\n[2/4] Prompt-Only Baseline...")
    prompt_results = evaluator.evaluate_prompt_only(
        args.trait, args.direction, eval_prompts
    )

    print("\n[3/4] Random Direction Baseline...")
    random_results = evaluator.evaluate_random_direction(
        hidden_dim, best_layer, args.alpha, eval_prompts
    )

    print("\n[4/4] Activation Steering (our method)...")
    steering_results = evaluator.evaluate_activation_steering(
        persona_vec, best_layer, args.alpha, eval_prompts
    )

    # Combine results
    all_results = {
        "model": args.model,
        "trait": args.trait,
        "direction": args.direction,
        "alpha": args.alpha,
        "best_layer": int(best_layer),
        "baselines": {
            "zero": zero_results,
            "prompt_only": prompt_results,
            "random_direction": random_results,
            "activation_steering": steering_results,
        },
    }

    # Save results
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f"baselines_{args.trait}_{args.direction}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_file}")
    print("\nNext steps:")
    print("1. Use BFI-44 or LLM-as-Judge to score these responses")
    print(
        "2. Compare scores across baselines to demonstrate activation steering superiority"
    )


if __name__ == "__main__":
    main()
