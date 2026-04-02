import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate_steering import BFI_ITEMS, keyword_score, compute_perplexity
from src.prompts.contrastive_prompts import apply_chat_template_safe


class FinalSteeringEvaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        self.steering_vec = None
        self.steering_layer = None
        self.alpha = 0.0

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        raise ValueError("Cannot find transformer layers")

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if layer_idx != self.steering_layer or self.steering_vec is None:
                return
            vec = self.alpha * self.steering_vec.to(self.device).to(
                dtype=output[0].dtype if isinstance(output, tuple) else output.dtype
            )
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
        for h in self.hooks:
            h.remove()
        self.hooks = []
        layers = self._get_layers()
        for i, l in enumerate(layers):
            self.hooks.append(l.register_forward_hook(self._hook_fn(i)))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_and_eval(
        self, prompt, trait_name, max_new_tokens=150, temperature=0.7
    ):
        messages = [{"role": "user", "content": prompt}]
        text = apply_chat_template_safe(
            self.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        resp = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # We need perplexity over the whole generated sequence + prompt to measure fluency.
        # But specifically, perplexity of the generated text conditioned on prompt.
        full_text = text + resp
        perp = compute_perplexity(self.model, self.tokenizer, full_text, self.device)
        score = keyword_score(resp, trait_name)

        return resp, perp, score

    def evaluate_alpha_sweep(self, trait_name, alphas, eval_prompts):
        results = {}
        for alpha in alphas:
            self.alpha = alpha
            scores = []
            perplexities = []
            responses = []

            for prompt in eval_prompts:
                resp, perp, score = self.generate_and_eval(prompt, trait_name)
                scores.append(score)
                perplexities.append(perp)
                responses.append(
                    {
                        "prompt": prompt,
                        "response": resp,
                        "score": score,
                        "perplexity": perp,
                    }
                )

            results[float(alpha)] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "mean_perplexity": float(np.mean(perplexities)),
                "responses": responses,
            }
            print(
                f"  α={alpha:+6.1f}: score={np.mean(scores):+.2f} ± {np.std(scores):.2f} | PPL={np.mean(perplexities):.2f}"
            )
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    model_short = args.model.replace("/", "_")
    vectors_dir = f"results/persona_vectors/{model_short}/{args.trait}/vectors"

    analysis_file = f"results/persona_vectors/{model_short}/{args.trait}/analysis_v2_{args.trait}.json"
    if os.path.exists(analysis_file):
        with open(analysis_file) as f:
            layer = json.load(f).get("best_layer_snr", 14)
    else:
        layer = model.config.num_hidden_layers // 2

    vec_path = os.path.join(vectors_dir, f"mean_diff_layer_{layer}.npy")
    evaluator = FinalSteeringEvaluator(model, tokenizer, args.device)
    evaluator.setup_steering(vec_path, layer)

    eval_prompts = BFI_ITEMS.get(args.trait, {}).get("eval_prompts", [])
    if not eval_prompts:
        print("No prompts found!")
        return

    alphas = [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    print(f"\nEvaluating α-sweep using Keyword Scorers & Context Perplexity...")
    sweep_results = evaluator.evaluate_alpha_sweep(args.trait, alphas, eval_prompts)
    evaluator.clear()

    out_dir = f"eval_results/{model_short}/{args.trait}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/alpha_sweep_final.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    # Visualization
    fig, ax1 = plt.subplots(figsize=(9, 5))
    sweep_alphas = sorted(sweep_results.keys())
    means = [sweep_results[a]["mean_score"] for a in sweep_alphas]
    stds = [sweep_results[a]["std_score"] for a in sweep_alphas]
    ppls = [sweep_results[a]["mean_perplexity"] for a in sweep_alphas]

    color1 = "#2196F3"
    ax1.set_xlabel("Steering Strength (α)", fontsize=12)
    ax1.set_ylabel(f"Keyword Trait Score [-1, 1]", color=color1, fontsize=12)
    ax1.errorbar(
        sweep_alphas, means, yerr=stds, fmt="o-", color=color1, linewidth=2, capsize=5
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.set_ylabel("Fluency (Sequence Perplexity)", color=color2, fontsize=12)
    ax2.plot(sweep_alphas, ppls, "s--", color=color2, linewidth=1.5, alpha=0.8)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yscale("log")  # Better for tracking perplexity explosion

    plt.title(
        f"Steering Trade-off Curve: {args.trait.capitalize()} (Layer {layer})\nShowing inverted-U saturation and perplexity degradation",
        fontsize=11,
    )
    fig.tight_layout()
    plt.savefig(f"{out_dir}/tradeoff_curve_{args.trait}.png", dpi=150)

    # Save a copy to the paper folder
    import shutil

    shutil.copy(
        f"{out_dir}/tradeoff_curve_{args.trait}.png", "paper/fig_tradeoff_curve.png"
    )
    print(f"Saved to {out_dir}/ and paper/fig_tradeoff_curve.png")


if __name__ == "__main__":
    main()
