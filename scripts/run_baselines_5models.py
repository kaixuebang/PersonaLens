import subprocess, os, sys

os.chdir("/data1/tongjizhou/persona")
PYTHON = "/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python"
env = os.environ.copy()
env["PYTHONPATH"] = "/data1/tongjizhou/persona"

MODELS = [
    "unsloth/llama-3-8B-Instruct",
    "unsloth/Llama-3.1-8B-Instruct",
    "/data0/shizitong/models/Phi3-mini-128k-instruct",
    "/data0/shizitong/models/Llama-2-7b-chat-hf",
    "/data0/shizitong/models/DeepSeek-R1-Distill-Qwen-14B",
]

for model in MODELS:
    ms = model.replace("/", "_")
    
    for script, label, results_dir in [
        ("src/evaluation/eval_shuffle_label_baseline.py", "shuffle-label", "results/shuffle_label_baseline_results"),
        ("src/evaluation/eval_null_orthogonality.py", "null-ortho", "results/null_orthogonality_results"),
    ]:
        out_dir = f"{results_dir}/{ms}"
        summary = f"{out_dir}/summary.json"
        if os.path.exists(summary):
            print(f"SKIP: {ms}/{label} (exists)")
            continue
        
        print(f"\n{'='*50}")
        print(f"{label}: {ms}")
        print(f"{'='*50}")
        
        cmd = [PYTHON, script, "--model", model]
        if "shuffle" in script:
            cmd += ["--n_permutations", "100"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-300:]}")
        else:
            print(result.stdout[-200:])
            print(f"  DONE")

print("\nAll baselines complete!")
