#!/usr/bin/env python3
"""
Score baseline comparison outputs using BFI-44
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse

# BFI-44 Items for Openness
BFI_44_OPENNESS = [
    {"item": "I am original, come up with new ideas", "keying": "positive"},
    {"item": "I am curious about many different things", "keying": "positive"},
    {"item": "I am ingenious, a deep thinker", "keying": "positive"},
    {"item": "I have an active imagination", "keying": "positive"},
    {"item": "I am inventive", "keying": "positive"},
    {"item": "I value artistic, aesthetic experiences", "keying": "positive"},
    {"item": "I prefer work that is routine", "keying": "negative"},
    {"item": "I like to reflect, play with ideas", "keying": "positive"},
    {"item": "I have few artistic interests", "keying": "negative"},
    {"item": "I prefer work that is routine", "keying": "negative"},
]


def score_with_bfi(model, tokenizer, text, device="cuda"):
    """
    Score a text using BFI-44 self-report methodology.
    Ask the model to rate itself on each BFI item based on the text.
    """
    scores = []

    for bfi_item in BFI_44_OPENNESS:
        item_text = bfi_item["item"]
        keying = bfi_item["keying"]

        prompt = f"""Based on the following self-description, please rate how much this person agrees with the statement.

Self-description: "{text[:500]}"

Statement: "{item_text}"

Please respond with a single number from 1 to 5:
1 = Strongly Disagree
2 = Disagree  
3 = Neutral
4 = Agree
5 = Strongly Agree

Rating (1-5):"""

        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # Extract numeric rating
            import re

            numbers = re.findall(r"\d+", response.strip())
            if numbers:
                rating = int(numbers[0])
                rating = max(1, min(5, rating))  # Clamp to 1-5

                # Reverse score for negative items
                if keying == "negative":
                    rating = 6 - rating

                scores.append(rating)
        except Exception as e:
            print(f"Error scoring item '{item_text}': {e}")
            scores.append(3)  # Default to neutral

    return sum(scores) / len(scores) if scores else 3.0


def score_baseline_file(baseline_file, model_name="Qwen/Qwen3-0.6B", device="cuda"):
    """Score all baselines in a file using BFI-44."""

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading baseline file: {baseline_file}")
    with open(baseline_file, "r") as f:
        data = json.load(f)

    results = {}

    for method_name, examples in data["baselines"].items():
        print(f"\n{'=' * 60}")
        print(f"Scoring method: {method_name}")
        print(f"{'=' * 60}")

        method_scores = []

        for i, example in enumerate(examples):
            response = example["response"]
            prompt = example["prompt"]

            # Extract just the response part (after </think> if present)
            if "</think>" in response:
                response_text = response.split("</think>")[-1].strip()
            else:
                response_text = response.strip()

            print(f"\nExample {i + 1}/{len(examples)}: {prompt[:50]}...")

            # Score with BFI-44
            bfi_score = score_with_bfi(model, tokenizer, response_text, device)
            method_scores.append(bfi_score)

            print(f"  BFI-44 Openness Score: {bfi_score:.2f}")

        mean_score = sum(method_scores) / len(method_scores)
        results[method_name] = {
            "individual_scores": method_scores,
            "mean_score": mean_score,
            "examples": len(examples),
        }

        print(f"\n{method_name.upper()} Mean BFI Score: {mean_score:.2f}")

    # Save results
    output_file = baseline_file.replace(".json", "_bfi_scored.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for method, result in results.items():
        print(f"{method:20s}: {result['mean_score']:.2f}")

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_file", required=True, help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Model to use for scoring"
    )
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    score_baseline_file(args.baseline_file, args.model, args.device)
