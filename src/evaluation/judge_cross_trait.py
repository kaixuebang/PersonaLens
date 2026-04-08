import json
import os
import re
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.judge_bfi_behavioral import build_judge_prompt, extract_rating, BIG_FIVE_TRAITS

BFI_DIR = Path("results/bfi_behavioral_v2")
ADJ_DIR = Path("results/bfi_adjusted_alpha")
CROSS_TRAIT_DIR = Path("results/cross_trait_interference")


def score_response(judge_model, judge_tokenizer, response_text, trait_name, device):
    prompt = build_judge_prompt(response_text, trait_name)
    messages = [{"role": "user", "content": prompt}]
    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs, max_new_tokens=100, temperature=0.1,
            do_sample=True, pad_token_id=judge_tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    judge_response = judge_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    rating = extract_rating(judge_response)
    return {"rating": rating, "judge_response": judge_response}


def process_cross_trait(responses_path, judge_model, judge_tokenizer, device,
                        force_overwrite=False):
    data = json.load(open(responses_path, encoding="utf-8"))
    steered_trait = data["trait"]
    model_name = data.get("model", "unknown")

    cross_trait_scores = {}

    for target_trait in BIG_FIVE_TRAITS:
        trait_scores = []
        score_key = f"judge_rating_{target_trait}"

        already_scored = 0
        need_scoring = 0
        for alpha_key, alpha_data in data["results"].items():
            for entry in alpha_data["scenario_results"]:
                if score_key in entry:
                    already_scored += 1
                else:
                    need_scoring += 1

        if need_scoring == 0 and not force_overwrite:
            print(f"    {target_trait}: already scored ({already_scored})")
            for alpha_key, alpha_data in data["results"].items():
                for entry in alpha_data["scenario_results"]:
                    trait_scores.append(entry[score_key])
            if trait_scores:
                cross_trait_scores[target_trait] = {
                    "mean": float(np.mean(trait_scores)),
                    "std": float(np.std(trait_scores)),
                    "count": len(trait_scores),
                }
            continue

        print(f"    {target_trait}: scoring {need_scoring} responses (skip {already_scored} existing)...")

        for alpha_key in sorted(data["results"].keys(), key=lambda x: float(x)):
            alpha_data = data["results"][alpha_key]
            alpha_val = alpha_data["alpha"]

            for entry in alpha_data["scenario_results"]:
                if score_key in entry and not force_overwrite:
                    trait_scores.append(entry[score_key])
                    continue

                result = score_response(
                    judge_model, judge_tokenizer,
                    entry["response"], target_trait, device
                )
                entry[score_key] = result["rating"]
                entry[f"judge_response_{target_trait}"] = result["judge_response"]
                trait_scores.append(result["rating"])

        if trait_scores:
            cross_trait_scores[target_trait] = {
                "mean": float(np.mean(trait_scores)),
                "std": float(np.std(trait_scores)),
                "count": len(trait_scores),
            }

    # Compute per-alpha cross-trait means
    alpha_cross_trait = {}
    for alpha_key in sorted(data["results"].keys(), key=lambda x: float(x)):
        alpha_val = data["results"][alpha_key]
        am = {}
        for target_trait in BIG_FIVE_TRAITS:
            sk = f"judge_rating_{target_trait}"
            ratings = [e[sk] for e in alpha_val["scenario_results"] if sk in e]
            if ratings:
                am[target_trait] = float(np.mean(ratings))
        alpha_cross_trait[str(alpha_val["alpha"])] = am

    cross_trait_scores["per_alpha_means"] = alpha_cross_trait
    data["cross_trait_interference"] = cross_trait_scores

    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved cross-trait scores for {steered_trait}")
    return cross_trait_scores


def main():
    parser = argparse.ArgumentParser(description="Cross-trait interference: judge all responses for all 5 traits")
    parser.add_argument("--judge", type=str, default="unsloth/gemma-2-2b-it")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trait", type=str, default=None)
    parser.add_argument("--source", choices=["original", "adjusted", "both"], default="both")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading judge: {args.judge}")

    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge, dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None, trust_remote_code=True,
    )
    if device != "cuda":
        judge_model = judge_model.to(device)
    judge_model.eval()

    source_dirs = []
    if args.source in ("original", "both"):
        source_dirs.append(("original", BFI_DIR))
    if args.source in ("adjusted", "both"):
        source_dirs.append(("adjusted", ADJ_DIR))

    for source_name, source_dir in source_dirs:
        model_dirs = sorted(d for d in source_dir.iterdir() if d.is_dir())
        for model_dir in model_dirs:
            model_short = model_dir.name
            if args.model and model_short != args.model.replace("/", "_"):
                continue

            for trait in BIG_FIVE_TRAITS:
                if args.trait and trait != args.trait:
                    continue

                responses_file = model_dir / f"responses_{trait}.json"
                if not responses_file.exists():
                    continue

                print(f"\n{'='*60}")
                print(f"  [{source_name}] {model_short} / {trait}")
                print(f"{'='*60}")

                process_cross_trait(
                    str(responses_file), judge_model, judge_tokenizer,
                    device, force_overwrite=args.force,
                )

    del judge_model, judge_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
