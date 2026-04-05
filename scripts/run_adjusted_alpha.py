"""
Run BFI behavioral eval with adjusted alphas + judge scoring.

Target: normalize perturbation ratio across models by scaling alpha.
Original alpha=6 gives:
  Q2.5-7B: 19.7%  -> scale 5x  -> alpha_max=30   -> perturb ~99%
  Q2.5-1.5: 18.4% -> scale 5x  -> alpha_max=30   -> perturb ~92%
  Q3-0.6: 10.8%   -> scale 9x  -> alpha_max=54   -> perturb ~97%
  Gemma-2B: 4.3%  -> scale 23x -> alpha_max=138  -> perturb ~100%

Usage:
    python scripts/run_adjusted_alpha.py --gpu 0
    python scripts/run_adjusted_alpha.py --gpu 0 --model Qwen/Qwen2.5-7B-Instruct
    python scripts/run_adjusted_alpha.py --collect-only
"""

import argparse
import json
import os
import subprocess
import sys

PYTHON = "/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python"
REPO = "/data1/tongjizhou/persona"

MODELS_CONFIG = [
    {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "alphas": [-30, -20, -10, 0, 10, 20, 30],
        "vram_gb": 14,
    },
    {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "alphas": [-30, -20, -10, 0, 10, 20, 30],
        "vram_gb": 4,
    },
    {
        "model": "Qwen/Qwen3-0.6B",
        "alphas": [-54, -36, -18, 0, 18, 36, 54],
        "vram_gb": 2,
    },
    {
        "model": "unsloth/gemma-2-2b-it",
        "alphas": [-138, -92, -46, 0, 46, 92, 138],
        "vram_gb": 6,
    },
]

OUTPUT_DIR = os.path.join(REPO, "results", "bfi_adjusted_alpha")


def run_bfi_eval(model_name, alphas, gpu):
    alpha_str = ",".join(str(a) for a in alphas)
    model_short = model_name.replace("/", "_")
    log_file = os.path.join(OUTPUT_DIR, f"bfi_{model_short}.log")

    cmd = [
        PYTHON, "-u", "-m", "src.evaluation.eval_bfi_behavioral_v2",
        "--model", model_name,
        "--trait", "all",
        "--output_dir", OUTPUT_DIR,
        f"--alphas={alpha_str}",
    ]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = REPO

    print(f"\n{'='*70}")
    print(f"  BFI Eval: {model_name}")
    print(f"  GPU: {gpu}, Alphas: {alpha_str}")
    print(f"  Log: {log_file}")
    print(f"{'='*70}")

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO
        )
    return proc, log_file


def run_judge_scoring(model_short, gpu):
    log_file = os.path.join(OUTPUT_DIR, f"judge_{model_short}.log")
    responses_dir = os.path.join(OUTPUT_DIR, model_short)

    cmd = [
        PYTHON, "-u", "-c", f"""
import sys, os, json, torch, numpy as np
from pathlib import Path
sys.path.insert(0, "{REPO}")
os.chdir("{REPO}")
from src.evaluation.judge_bfi_behavioral import process_file
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
print("Loading judge model: {OUTPUT_DIR}")
judge_tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b-it", trust_remote_code=True)
judge_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-2-2b-it", dtype=torch.float16, device_map="cuda", trust_remote_code=True
)
judge_model.eval()

responses_dir = Path("{responses_dir}")
traits = ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness"]
for trait in traits:
    fpath = responses_dir / f"responses_{{trait}}.json"
    if fpath.exists():
        print(f"  Judging {{trait}}...")
        process_file(str(fpath), judge_model, judge_tokenizer, trait, device)
    else:
        print(f"  SKIP {{trait}}: not found")

del judge_model, judge_tokenizer
torch.cuda.empty_cache()
print("Done judging {model_short}")
""",
    ]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = REPO

    print(f"  Judge scoring: {model_short}...")

    with open(log_file, "w") as log_f:
        result = subprocess.run(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO
        )
        if result.returncode != 0:
            print(f"  WARNING: Judge failed (exit {result.returncode}). See {log_file}")
        else:
            print(f"  Judge done for {model_short}")


def collect_results():
    import numpy as np

    print(f"\n{'='*70}")
    print("ADJUSTED ALPHA RESULTS SUMMARY")
    print(f"{'='*70}")

    for config in MODELS_CONFIG:
        model_short = config["model"].replace("/", "_")
        model_dir = os.path.join(OUTPUT_DIR, model_short)

        if not os.path.isdir(model_dir):
            continue

        print(f"\n--- {model_short} (alphas={config['alphas']}) ---")

        traits_data = {}
        for trait in ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness"]:
            fpath = os.path.join(model_dir, f"responses_{trait}.json")
            if not os.path.exists(fpath):
                continue

            with open(fpath) as f:
                d = json.load(f)

            results = d.get("results", {})
            alpha_scores = {}
            for k, v in results.items():
                if isinstance(v, dict) and "judge_mean" in v:
                    try:
                        alpha_scores[float(k)] = v["judge_mean"]
                    except (ValueError, TypeError):
                        pass

            if alpha_scores:
                sorted_alphas = sorted(alpha_scores.keys())
                scores = [alpha_scores[a] for a in sorted_alphas]
                delta = max(scores) - min(scores)
                monotonic = np.corrcoef(sorted_alphas, scores)[0, 1] if len(sorted_alphas) > 2 else 0
                traits_data[trait] = {"delta": delta, "monotonic": monotonic}

                print(f"  {trait[:5]:>5}: delta={delta:+.2f}, r={monotonic:.2f}")

        if traits_data:
            mean_delta = np.mean([v["delta"] for v in traits_data.values()])
            mean_mono = np.mean([v["monotonic"] for v in traits_data.values()])
            print(f"  >> Mean delta={mean_delta:.2f}, Mean r={mean_mono:.2f}")

    print(f"\n{'='*70}")
    print("COMPARISON: Original (alpha=[-6,6]) vs Adjusted")
    print(f"{'='*70}")
    orig_dir = os.path.join(REPO, "results", "bfi_behavioral_v2")

    for config in MODELS_CONFIG:
        model_short = config["model"].replace("/", "_")
        orig_deltas = []
        adj_deltas = []

        for trait in ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness"]:
            for label, dir_path, deltas in [("orig", orig_dir, orig_deltas), ("adj", OUTPUT_DIR, adj_deltas)]:
                fpath = os.path.join(dir_path, model_short, f"responses_{trait}.json")
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        d = json.load(f)
                    results = d.get("results", {})
                    alpha_scores = {}
                    for k, v in results.items():
                        if isinstance(v, dict) and "judge_mean" in v:
                            try:
                                alpha_scores[float(k)] = v["judge_mean"]
                            except:
                                pass
                    if alpha_scores:
                        vals = list(alpha_scores.values())
                        deltas.append(max(vals) - min(vals))

        orig_mean = np.mean(orig_deltas) if orig_deltas else 0
        adj_mean = np.mean(adj_deltas) if adj_deltas else 0
        change = adj_mean - orig_mean
        arrow = "UP" if change > 0.05 else "DOWN" if change < -0.05 else "SAME"
        print(f"  {model_short:<35} Orig={orig_mean:.2f}  Adj={adj_mean:.2f}  {arrow} ({change:+.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--skip-bfi", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.collect_only:
        collect_results()
        return

    models = MODELS_CONFIG
    if args.model:
        models = [m for m in models if m["model"] == args.model]
        if not models:
            print(f"ERROR: Model {args.model} not found in config")
            sys.exit(1)

    if not args.skip_bfi:
        for config in models:
            proc, log_file = run_bfi_eval(config["model"], config["alphas"], args.gpu)
            print(f"  Waiting for {config['model']}... (PID={proc.pid})")
            proc.wait()
            print(f"  Done. Exit code: {proc.returncode}")

    if not args.skip_judge:
        print(f"\n{'='*70}")
        print("Phase 2: LLM-as-Judge Scoring")
        print(f"{'='*70}")

        for config in models:
            model_short = config["model"].replace("/", "_")
            responses_dir = os.path.join(OUTPUT_DIR, model_short)
            if os.path.isdir(responses_dir):
                run_judge_scoring(model_short, args.gpu)

    collect_results()


if __name__ == "__main__":
    main()
