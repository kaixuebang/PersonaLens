"""
Re-run Llama-3-8B and Llama-3.1-8B with standardized V2 protocol:
  - 7 scenarios × 5 reps × 7 alphas = 245 responses per trait
  - Uses DEFAULT_ALPHAS = [-6, -4, -2, 0, 2, 4, 6]
  - Then judges with Gemma-2-2B
"""
import subprocess, sys, os

os.chdir("/data1/tongjizhou/persona")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PYTHON = "/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python"
GPU = sys.argv[1] if len(sys.argv) > 1 else "2"

MODELS = [
    ("unsloth/llama-3-8B-Instruct", "unsloth_llama-3-8B-Instruct"),
    ("unsloth/Llama-3.1-8B-Instruct", "unsloth_Llama-3.1-8B-Instruct"),
]

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

for model_name, model_short in MODELS:
    print(f"\n{'='*60}")
    print(f"PHASE 1: Steering evaluation for {model_name}")
    print(f"{'='*60}")
    
    # Backup old results
    import shutil
    old_dir = f"results/bfi_behavioral_v2/{model_short}"
    backup_dir = f"results/bfi_behavioral_v2/{model_short}_v1_backup"
    if not os.path.exists(backup_dir) and os.path.exists(old_dir):
        os.makedirs(backup_dir, exist_ok=True)
        for t in TRAITS:
            src = f"{old_dir}/responses_{t}.json"
            if os.path.exists(src):
                shutil.copy2(src, f"{backup_dir}/responses_{t}.json")
        print(f"  Backed up old results to {backup_dir}")
    
    # Run steering eval with V2 protocol
    cmd = [
        PYTHON, "-m", "src.evaluation.eval_bfi_behavioral_v2",
        "--model", model_name,
        "--trait", "all",
        "--device", "cuda" if GPU in ["0","1","2","3","4","5","6","7"] else f"cuda:{GPU}",
        "--output_dir", "results/bfi_behavioral_v2",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    
    print(f"  Running: {' '.join(cmd)}")
    print(f"  GPU: {GPU}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        continue
    print(result.stdout[-500:])

print(f"\n{'='*60}")
print(f"PHASE 2: Judging all responses with Gemma-2-2B")
print(f"{'='*60}")

cmd = [
    PYTHON, "-m", "src.evaluation.judge_bfi_behavioral",
    "--judge", "unsloth/gemma-2-2b-it",
    "--device", "cuda" if GPU in ["0","1","2","3","4","5","6","7"] else f"cuda:{GPU}",
]
env["CUDA_VISIBLE_DEVICES"] = GPU
result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
if result.returncode != 0:
    print(f"  ERROR: {result.stderr[-500:]}")
else:
    print(result.stdout[-500:])

# Phase 3: Cross-trait judging
print(f"\n{'='*60}")
print(f"PHASE 3: Cross-trait judging")
print(f"{'='*60}")

for model_name, model_short in MODELS:
    cmd = [
        PYTHON, "-m", "src.evaluation.judge_cross_trait",
        "--model", model_short,
        "--device", "cuda" if GPU in ["0","1","2","3","4","5","6","7"] else f"cuda:{GPU}",
    ]
    env["CUDA_VISIBLE_DEVICES"] = GPU
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if result.returncode != 0:
        print(f"  {model_short}: ERROR {result.stderr[-200:]}")
    else:
        print(f"  {model_short}: Done")

print("\nAll phases complete!")
