import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate_steering import BFI_ITEMS, compute_perplexity

class SelfReportEvaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        self.steering_vec = None
        self.steering_layer = None
        self.alpha = 0.0

    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        raise ValueError("Cannot find transformer layers")

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if layer_idx != self.steering_layer or self.steering_vec is None:
                return
            vec = self.alpha * self.steering_vec.to(self.device).to(dtype=output[0].dtype if isinstance(output, tuple) else output.dtype)
            if isinstance(output, tuple):
                mod = list(output)
                mod[0] = output[0] + vec.unsqueeze(0).unsqueeze(0)
                return tuple(mod)
            return output + vec.unsqueeze(0).unsqueeze(0)
        return hook

    def setup_steering(self, vec_path, layer_idx, dtype=torch.float16):
        vec = np.load(vec_path)
        self.steering_vec = torch.tensor(vec, dtype=dtype)
        self.steering_layer = layer_idx
        for h in self.hooks: h.remove()
        self.hooks = []
        layers = self._get_layers()
        for i, l in enumerate(layers):
            self.hooks.append(l.register_forward_hook(self._hook_fn(i)))

    def clear(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def get_self_report_score(self, stmt, is_positive):
        # We ask the model to rate the statement 1-5
        prompt = f"Please rate how much you agree with the following statement about yourself from 1 (strongly disagree) to 5 (strongly agree). Respond with ONLY a single digit integer (1, 2, 3, 4, or 5).\n\nStatement: \"{stmt}\"\n\nRating (1-5):"
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # We look at the logits for '1', '2', '3', '4', '5' for the immediate next token
        with torch.no_grad():
            out = self.model(**inputs)
        
        next_token_logits = out.logits[0, -1, :]
        
        # Qwen encodes numbers directly or with spaces. Let's dynamically get the token ids for 1, 2, 3, 4, 5.
        token_ids = [self.tokenizer.encode(str(i))[-1] for i in range(1, 6)]
        
        # Extract logits and apply softmax
        rating_logits = next_token_logits[token_ids]
        probs = torch.softmax(rating_logits, dim=-1)
        
        # Expected value
        expected_rating = sum((i+1) * probs[i].item() for i in range(5))
        
        # Invert if negative
        if not is_positive:
            expected_rating = 6.0 - expected_rating
            
        return expected_rating

    def evaluate_alpha_sweep(self, alphas, items):
        results = {}
        for alpha in alphas:
            self.alpha = alpha
            scores = []
            responses = []
            for stmt, is_pos in items:
                score = self.get_self_report_score(stmt, is_pos)
                scores.append(score)
                responses.append({"statement": stmt, "is_positive": is_pos, "expected_rating": score})

            results[float(alpha)] = {
                "mean_rating": float(np.mean(scores)),
                "std_rating": float(np.std(scores)),
                "responses": responses,
            }
            print(f"  α={alpha:+6.1f}: Mean BFI Score={np.mean(scores):.2f} ± {np.std(scores):.2f}")
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True)
    model.eval()

    model_short = args.model.replace("/", "_")
    vectors_dir = f"persona_vectors_v2/{model_short}/{args.trait}/vectors"
    
    analysis_file = f"persona_vectors_v2/{model_short}/{args.trait}/analysis_v2_{args.trait}.json"
    if os.path.exists(analysis_file):
        with open(analysis_file) as f:
            layer = json.load(f).get("best_layer_snr", 14)
    else:
        layer = model.config.num_hidden_layers // 2

    vec_path = os.path.join(vectors_dir, f"mean_diff_layer_{layer}.npy")
    evaluator = SelfReportEvaluator(model, tokenizer, args.device)
    evaluator.setup_steering(vec_path, layer)

    # Gather items
    trait_dict = BFI_ITEMS.get(args.trait, {})
    pos_items = [(x, True) for x in trait_dict.get("positive", [])]
    neg_items = [(x, False) for x in trait_dict.get("negative", [])]
    all_items = pos_items + neg_items
    
    if not all_items:
        print(f"No BFI items defined for {args.trait}")
        return

    alphas = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    
    print(f"\nEvaluating α-sweep using BFI Logistic Self-Reporting...")
    sweep_results = evaluator.evaluate_alpha_sweep(alphas, all_items)
    evaluator.clear()

    out_dir = f"eval_results_v2/{model_short}/{args.trait}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/bfi_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    sweep_alphas = sorted(sweep_results.keys())
    means = [sweep_results[a]["mean_rating"] for a in sweep_alphas]
    stds = [sweep_results[a]["std_rating"] for a in sweep_alphas]

    color = '#2196F3'
    ax1.set_xlabel('Steering Strength (α)', fontsize=12)
    ax1.set_ylabel(f'BFI Self-Reported Trait Score [1-5]', color=color, fontsize=12)
    ax1.errorbar(sweep_alphas, means, yerr=stds, fmt='o-', color=color, linewidth=2, capsize=5)
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax1.axhline(y=3.0, color="gray", linestyle="--", alpha=0.3)
    ax1.set_ylim(1.0, 5.0)

    plt.title(f"Quantitative Steering: {args.trait.capitalize()} (Layer {layer})\nEvaluated via logit-based BFI Self-Reporting", fontsize=11)
    fig.tight_layout()
    plt.savefig(f"{out_dir}/alpha_curve_bfi_{args.trait}.png", dpi=150)
    print(f"Saved to {out_dir}/")

if __name__ == "__main__":
    main()
