"""
Phase 2: LLM-as-Judge Scoring for Behavioral BFI Responses

Loads raw responses from eval_bfi_behavioral_v2.py and scores each one
using a judge model (Gemma-2-2B by default) with structured BFI rubrics.

The judge receives the response text + trait rubric and outputs a 1-5 rating
with reasoning. We extract the numerical rating for dose-response analysis.

Usage:
    python -m src.evaluation.judge_bfi_behavioral --judge unsloth/gemma-2-2b-it
    python -m src.evaluation.judge_bfi_behavioral --judge unsloth/gemma-2-2b-it --model Qwen/Qwen3-0.6B --trait openness
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.bfi_44_scale import BFI_JUDGE_RUBRICS

BIG_FIVE_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
RESPONSES_DIR = "results/bfi_behavioral_v2"
OUTPUT_DIR = "results/bfi_behavioral_v2"


def build_judge_prompt(response_text, trait_name):
    rubric = BFI_JUDGE_RUBRICS[trait_name]
    high = "\n".join(f"  - {h}" for h in rubric["high_indicators"])
    low = "\n".join(f"  - {l}" for l in rubric["low_indicators"])

    return f"""You are a psychologist evaluating a person's response for the personality trait "{trait_name.capitalize()}".

TRAIT: {trait_name.capitalize()}
{rubric["description"]}

HIGH {trait_name.upper()} BEHAVIORS:
{high}

LOW {trait_name.upper()} BEHAVIORS:
{low}

PERSON'S RESPONSE:
\"\"\"
{response_text}
\"\"\"

Rate how much this response reflects {trait_name.capitalize()} on a 1-5 scale:
1 = Very Low (clearly shows LOW indicators)
2 = Low (mostly shows LOW indicators)
3 = Moderate (mixed or neutral)
4 = High (mostly shows HIGH indicators)
5 = Very High (clearly shows HIGH indicators)

First write one sentence of reasoning, then output your rating on a new line.
Format: RATING: [1-5]"""


def extract_rating(text):
    match = re.search(r"RATING:\s*(\d)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:rating|score)\s*(?:is|=|:)\s*(\d)", text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 5:
            return val
    nums = re.findall(r"\b([1-5])\b", text[-50:])
    if nums:
        return int(nums[-1])
    return 3


def score_response(judge_model, judge_tokenizer, response_text, trait_name, device):
    prompt = build_judge_prompt(response_text, trait_name)
    messages = [{"role": "user", "content": prompt}]
    text = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    judge_response = judge_tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()
    rating = extract_rating(judge_response)

    return {
        "rating": rating,
        "judge_response": judge_response,
    }


def process_file(responses_path, judge_model, judge_tokenizer, trait, device):
    data = json.load(open(responses_path, encoding="utf-8"))
    print(f"  Processing {responses_path} ({len(data['alphas'])} alphas)...")

    for alpha_key, alpha_data in data["results"].items():
        scored_count = 0
        for entry in alpha_data["scenario_results"]:
            if "judge_rating" in entry:
                scored_count += 1
                continue

            result = score_response(
                judge_model, judge_tokenizer, entry["response"], trait, device
            )
            entry["judge_rating"] = result["rating"]
            entry["judge_response"] = result["judge_response"]

        alpha_ratings = [
            e["judge_rating"]
            for e in alpha_data["scenario_results"]
            if "judge_rating" in e
        ]
        if alpha_ratings:
            alpha_data["judge_mean"] = float(np.mean(alpha_ratings))
            alpha_data["judge_std"] = float(np.std(alpha_ratings))
            alpha_data["judge_ci_lower"] = float(np.percentile(alpha_ratings, 2.5))
            alpha_data["judge_ci_upper"] = float(np.percentile(alpha_ratings, 97.5))
            print(
                f"    α={alpha_data['alpha']:+5.1f}: {len(alpha_ratings)} rated, "
                f"mean={alpha_data['judge_mean']:.2f} ± {alpha_data['judge_std']:.2f}"
            )

    dose_response = {
        "alphas": data["alphas"],
        "means": [],
        "stds": [],
        "ci_lowers": [],
        "ci_uppers": [],
    }
    for a in data["alphas"]:
        k = str(float(a))
        if k in data["results"] and "judge_mean" in data["results"][k]:
            dose_response["means"].append(data["results"][k]["judge_mean"])
            dose_response["stds"].append(data["results"][k]["judge_std"])
            dose_response["ci_lowers"].append(data["results"][k]["judge_ci_lower"])
            dose_response["ci_uppers"].append(data["results"][k]["judge_ci_upper"])
        else:
            dose_response["means"].append(None)
            dose_response["stds"].append(None)
            dose_response["ci_lowers"].append(None)
            dose_response["ci_uppers"].append(None)

    data["dose_response_judge"] = dose_response

    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Updated {responses_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge scoring for behavioral BFI"
    )
    parser.add_argument("--judge", type=str, default="unsloth/gemma-2-2b-it")
    parser.add_argument(
        "--model", type=str, default=None, help="Filter to specific source model"
    )
    parser.add_argument(
        "--trait", type=str, default=None, help="Filter to specific trait"
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading judge model: {args.judge}")

    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        judge_model = judge_model.to(device)
    judge_model.eval()

    responses_root = Path(RESPONSES_DIR)
    model_dirs = sorted(d for d in responses_root.iterdir() if d.is_dir())

    for model_dir in model_dirs:
        model_short = model_dir.name
        if args.model and model_short != args.model.replace("/", "_"):
            continue

        for trait in BIG_FIVE_TRAITS:
            if args.trait and trait != args.trait:
                continue

            responses_file = model_dir / f"responses_{trait}.json"
            if not responses_file.exists():
                print(f"  SKIP: {responses_file} not found")
                continue

            print(f"\n{'=' * 50}")
            print(f"  Judging: {model_short} / {trait}")
            print(f"{'=' * 50}")
            process_file(
                str(responses_file), judge_model, judge_tokenizer, trait, device
            )

    del judge_model, judge_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
