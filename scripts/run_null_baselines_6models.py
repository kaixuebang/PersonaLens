"""Run shuffle-label and null-orthogonality baselines for the 6 new models."""
import subprocess, sys, os

os.chdir("/data1/tongjizhou/persona")
PYTHON = "/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python"

MODELS = [
    "unsloth/llama-3-8B-Instruct",
    "unsloth/Llama-3.1-8B-Instruct",
    "/data0/shizitong/models/Phi3-mini-128k-instruct",
    "/data0/shizitong/models/Llama-2-7b-chat-hf",
    "/data0/shizitong/models/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/Qwen2.5-14B-Instruct",
]

for model in MODELS:
    ms = model.replace("/", "_")
    
    # Shuffle-label baseline
    out_dir = f"results/shuffle_label_baseline_results/{ms}"
    if not os.path.exists(f"{out_dir}/summary.json"):
        print(f"\n{'='*50}")
        print(f"Shuffle-label baseline: {ms}")
        print(f"{'='*50}")
        cmd = [
            PYTHON, "src/evaluation/eval_shuffle_label_baseline.py",
            "--model", model, "--n_permutations", "100",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            print(f"  DONE")
    else:
        print(f"SKIP (exists): {ms} shuffle-label")
    
    # Null orthogonality
    out_dir2 = f"results/null_orthogonality_results/{ms}"
    if not os.path.exists(f"{out_dir2}/summary.json"):
        print(f"\n{'='*50}")
        print(f"Null orthogonality: {ms}")
        print(f"{'='*50}")
        cmd = [
            PYTHON, "src/evaluation/eval_null_orthogonality.py",
            "--model", model,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            print(f"  DONE")
    else:
        print(f"SKIP (exists): {ms} null-orthogonality")

print("\nAll baselines complete!")
