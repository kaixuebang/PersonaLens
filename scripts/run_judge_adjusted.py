"""
Run LLM-as-Judge scoring on adjusted-alpha BFI results.
Uses process_file() from judge_bfi_behavioral.py to score responses
in results/bfi_adjusted_alpha/.

Usage:
    python scripts/run_judge_adjusted.py --gpu 0
    python scripts/run_judge_adjusted.py --gpu 0 --model Qwen_Qwen3-0.6B --trait openness
"""

import argparse
import json
import os
import sys
import time

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "/data1/tongjizhou/persona"
ADJUSTED_DIR = os.path.join(REPO, "results", "bfi_adjusted_alpha")
JUDGE_MODEL = "unsloth/gemma-2-2b-it"
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

sys.path.insert(0, REPO)
os.chdir(REPO)

from src.evaluation.judge_bfi_behavioral import process_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default=None, help="Filter model dir name")
    parser.add_argument("--trait", type=str, default=None, help="Filter trait")
    args = parser.parse_args()

    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Loading judge model: {JUDGE_MODEL} on GPU {args.gpu}")
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    judge_model.eval()
    print("Judge model loaded.")

    adjusted_root = Path(ADJUSTED_DIR)
    model_dirs = sorted(d for d in adjusted_root.iterdir() if d.is_dir())

    total_start = time.time()

    for model_dir in model_dirs:
        model_short = model_dir.name
        if args.model and model_short != args.model:
            continue

        print(f"\n{'='*60}")
        print(f"  Judging: {model_short}")
        print(f"{'='*60}")

        for trait in TRAITS:
            if args.trait and trait != args.trait:
                continue

            responses_file = model_dir / f"responses_{trait}.json"
            if not responses_file.exists():
                print(f"  SKIP {trait}: file not found")
                continue

            print(f"\n  --- {trait} ---")
            process_file(
                str(responses_file), judge_model, judge_tokenizer, trait, device
            )

    elapsed = time.time() - total_start
    print(f"\nAll judging complete in {elapsed:.1f}s")

    del judge_model, judge_tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
