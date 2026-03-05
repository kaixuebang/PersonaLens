"""
Phase 5: Inference-Time Personality Steering

Demonstrates that injecting persona vectors into hidden states during
inference can control the model's personality expression WITHOUT
any prompt engineering or fine-tuning.

This is the key experiment: proving that controlling neuron activations
controls personality.

Usage:
    python steer_personality.py --model Qwen/Qwen3-0.6B --trait openness --alpha 3.0
    python steer_personality.py --model Qwen/Qwen3-0.6B --trait humor --alpha 5.0
    python steer_personality.py --model Qwen/Qwen3-0.6B --trait openness --sweep
"""

import argparse
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class PersonalitySteerer:
    """
    Steers model personality by adding persona vectors to hidden states
    during inference, following the methodology from RepE and ITI.

    h_l' = h_l + alpha * v_persona

    where v_persona is the extracted personality direction for layer l.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        self.steering_vectors = {}  # layer_idx -> (vector, alpha)
        self.active = False

    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers
        else:
            raise ValueError("Cannot find transformer layers")

    def _steering_hook(self, layer_idx):
        """Hook that adds the persona vector to the hidden state."""
        def hook_fn(module, input, output):
            if not self.active or layer_idx not in self.steering_vectors:
                return output

            vector, alpha = self.steering_vectors[layer_idx]
            steering = alpha * vector.to(self.device)

            if isinstance(output, tuple):
                modified = list(output)
                # Add steering vector to all token positions
                modified[0] = output[0] + steering.unsqueeze(0).unsqueeze(0)
                return tuple(modified)
            else:
                return output + steering.unsqueeze(0).unsqueeze(0)
        return hook_fn

    def load_persona_vectors(self, vectors_dir, layer_indices=None, alpha=1.0,
                              vector_type="mean_diff"):
        """
        Load persona vectors from the extraction phase.

        Args:
            vectors_dir: Directory containing extracted vectors
            layer_indices: Which layers to steer (None = best layer only)
            alpha: Steering strength
            vector_type: "mean_diff" or "probe_dir"
        """
        self.steering_vectors = {}

        if layer_indices is None:
            # Load analysis to find best layer
            parent_dir = os.path.dirname(vectors_dir)
            trait_name = os.path.basename(parent_dir)
            analysis_file = os.path.join(parent_dir, f"analysis_{trait_name}.json")
            if os.path.exists(analysis_file):
                with open(analysis_file) as f:
                    analysis = json.load(f)
                best_layer = analysis["best_layer"]
                layer_indices = [best_layer]
                print(f"  Auto-selected best layer: {best_layer} "
                      f"(probe acc: {analysis['best_probe_accuracy']:.3f})")
            else:
                # Default to middle layers
                n_layers = self.model.config.num_hidden_layers
                mid = n_layers // 2
                layer_indices = [mid - 1, mid, mid + 1]
                print(f"  Using default middle layers: {layer_indices}")

        for layer_idx in layer_indices:
            vec_file = os.path.join(vectors_dir, f"{vector_type}_layer_{layer_idx}.npy")
            if os.path.exists(vec_file):
                vec = np.load(vec_file)
                vec_tensor = torch.tensor(vec, dtype=torch.float16 if self.device == "cuda"
                                          else torch.float32)
                self.steering_vectors[layer_idx] = (vec_tensor, alpha)
                print(f"  Loaded {vector_type} vector for layer {layer_idx}, alpha={alpha}")
            else:
                print(f"  WARNING: Vector file not found: {vec_file}")

    def set_alpha(self, alpha):
        """Update steering strength for all loaded vectors."""
        for layer_idx in self.steering_vectors:
            vec, _ = self.steering_vectors[layer_idx]
            self.steering_vectors[layer_idx] = (vec, alpha)

    def register_hooks(self):
        """Register steering hooks on the model."""
        self._clear_hooks()
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._steering_hook(i))
            self.hooks.append(hook)
        self.active = True

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active = False

    def generate(self, prompt, max_new_tokens=200, temperature=0.7, do_sample=True,
                 steer=True):
        """
        Generate text with or without personality steering.

        Args:
            prompt: User message (string)
            steer: Whether to apply steering
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        if steer and self.steering_vectors:
            self.register_hooks()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        if steer:
            self._clear_hooks()

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response


def run_steering_comparison(steerer, prompts, alpha, max_new_tokens=200):
    """Run generation with and without steering for comparison."""
    results = []

    for prompt in prompts:
        # Baseline (no steering)
        steerer.active = False
        baseline_response = steerer.generate(prompt, max_new_tokens=max_new_tokens, steer=False)

        # Steered
        steerer.set_alpha(alpha)
        steered_response = steerer.generate(prompt, max_new_tokens=max_new_tokens, steer=True)

        # Negative steering (opposite direction)
        steerer.set_alpha(-alpha)
        neg_steered_response = steerer.generate(prompt, max_new_tokens=max_new_tokens, steer=True)

        results.append({
            "prompt": prompt,
            "baseline": baseline_response,
            "steered_positive": steered_response,
            "steered_negative": neg_steered_response,
            "alpha": alpha,
        })

    return results


def run_alpha_sweep(steerer, prompts, alphas, max_new_tokens=200):
    """Sweep over different steering strengths."""
    sweep_results = {}

    for alpha in alphas:
        print(f"\n  Alpha = {alpha}")
        steerer.set_alpha(alpha)
        responses = []
        for prompt in prompts:
            resp = steerer.generate(prompt, max_new_tokens=max_new_tokens, steer=True)
            responses.append({"prompt": prompt, "response": resp})
        sweep_results[alpha] = responses

    return sweep_results


# Default evaluation prompts (neutral - no personality priming)
EVAL_PROMPTS = [
    "What do you think about trying a completely new hobby you've never considered before?",
    "How would you react if someone criticized your work in front of others?",
    "Tell me about how you handle a stressful deadline.",
    "What's your approach to meeting new people at a social event?",
    "How do you feel about taking risks?",
    "Describe how you deal with a major setback in life.",
    "What would you do if you found out a friend lied to you?",
    "How do you spend your time when you're feeling down?",
    "What's your opinion on always following rules versus sometimes bending them?",
    "How do you handle uncertainty in your life?",
]


def main():
    parser = argparse.ArgumentParser(description="Inference-time personality steering")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trait", type=str, required=True,
                        help="Trait to steer (e.g., openness, humor)")
    parser.add_argument("--vectors_dir", type=str, default=None,
                        help="Directory with extracted persona vectors. "
                             "Default: persona_vectors/<model>/<trait>/vectors/")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="Steering strength")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to steer (default: auto)")
    parser.add_argument("--vector_type", type=str, default="mean_diff",
                        choices=["mean_diff", "probe_dir"],
                        help="Which type of persona vector to use")
    parser.add_argument("--sweep", action="store_true",
                        help="Run alpha sweep instead of single comparison")
    parser.add_argument("--output_dir", type=str, default="steering_results")
    parser.add_argument("--max_new_tokens", type=int, default=200)
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

    # Setup steerer
    steerer = PersonalitySteerer(model, tokenizer, args.device)

    # Determine vectors directory
    if args.vectors_dir is None:
        model_short = args.model.replace("/", "_")
        args.vectors_dir = os.path.join("persona_vectors", model_short, args.trait, "vectors")

    # Parse layers
    layer_indices = None
    if args.layers:
        layer_indices = [int(l) for l in args.layers.split(",")]

    # Load persona vectors
    steerer.load_persona_vectors(
        args.vectors_dir,
        layer_indices=layer_indices,
        alpha=args.alpha,
        vector_type=args.vector_type,
    )

    if not steerer.steering_vectors:
        print("ERROR: No persona vectors loaded. Run extract_persona_vectors.py first.")
        return

    # Output directory
    model_short = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_short, args.trait)
    os.makedirs(output_dir, exist_ok=True)

    if args.sweep:
        # Alpha sweep
        alphas = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, -3.0, -5.0]
        print(f"\n{'='*60}")
        print(f"Running alpha sweep for {args.trait}: {alphas}")
        print(f"{'='*60}")

        sweep_results = run_alpha_sweep(steerer, EVAL_PROMPTS[:5], alphas, args.max_new_tokens)

        # Save results
        json_path = os.path.join(output_dir, f"alpha_sweep_{args.trait}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved sweep results to {json_path}")

        # Print selected examples
        for alpha_val in [0.0, 3.0, -3.0]:
            if alpha_val in sweep_results:
                print(f"\n--- Alpha = {alpha_val} ---")
                for item in sweep_results[alpha_val][:2]:
                    print(f"  Q: {item['prompt'][:80]}...")
                    print(f"  A: {item['response'][:200]}...")

    else:
        # Single comparison
        print(f"\n{'='*60}")
        print(f"Steering comparison for {args.trait} (alpha={args.alpha})")
        print(f"{'='*60}")

        results = run_steering_comparison(steerer, EVAL_PROMPTS, args.alpha, args.max_new_tokens)

        # Save results
        json_path = os.path.join(output_dir, f"comparison_{args.trait}_alpha{args.alpha}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print results
        for r in results:
            print(f"\n{'─'*60}")
            print(f"Prompt: {r['prompt']}")
            print(f"\n  [BASELINE]:\n  {r['baseline'][:300]}")
            print(f"\n  [+{args.trait.upper()} (α={args.alpha})]:\n  {r['steered_positive'][:300]}")
            print(f"\n  [-{args.trait.upper()} (α=-{args.alpha})]:\n  {r['steered_negative'][:300]}")

        print(f"\n✓ Results saved to {json_path}")

    print("\n✓ Steering experiment complete!")


if __name__ == "__main__":
    main()
