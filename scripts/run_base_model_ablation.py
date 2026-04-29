"""
Instruction-Tuning Ablation: Compare Llama-2-7b-hf (base) vs Llama-2-7b-chat-hf (instruct).
This addresses Limitation (4): Is instruction-tuning a key factor in Mode 2 failure?
"""
import subprocess, sys, os

os.chdir("/data1/tongjizhou/persona")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PYTHON = "/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python"
GPU = sys.argv[1] if len(sys.argv) > 1 else "4"

BASE_MODEL = "/data0/shizitong/models/Llama-2-7b-hf"
MODEL_SHORT = "Llama-2-7b-hf_BASE"
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPU

print(f"{'='*60}")
print(f"INSTRUCTION-TUNING ABLATION")
print(f"Base model: {BASE_MODEL}")
print(f"GPU: {GPU}")
print(f"{'='*60}")

# Phase 1: Extract persona vectors (with activations)
print(f"\n--- Phase 1: Activation collection + Vector extraction ---")
for trait in TRAITS:
    print(f"\n  Extracting {trait}...")
    cmd = [PYTHON, "-m", "src.extraction.extract_persona_vectors_v2",
           "--model", BASE_MODEL,
           "--trait", trait,
           "--device", "cuda",
           "--output_dir", "results/persona_vectors"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
    else:
        # Get LOSO accuracy from output
        for line in result.stdout.split('\n'):
            if 'LOSO' in line or 'accuracy' in line.lower() or 'Cohen' in line:
                print(f"    {line.strip()}")

# Phase 2: Steering evaluation
print(f"\n--- Phase 2: Steering evaluation ---")
cmd = [PYTHON, "-m", "src.evaluation.eval_bfi_behavioral_v2",
       "--model", BASE_MODEL,
       "--trait", "all",
       "--device", "cuda",
       "--output_dir", "results/bfi_behavioral_v2"]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
if result.returncode != 0:
    print(f"  ERROR: {result.stderr[-300:]}")
else:
    print(result.stdout[-500:])

# Phase 3: Judge
print(f"\n--- Phase 3: Judging ---")
cmd = [PYTHON, "-m", "src.evaluation.judge_bfi_behavioral",
       "--model", BASE_MODEL.replace("/", "_"),
       "--trait", "all",
       "--device", "cuda"]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
if result.returncode != 0:
    print(f"  ERROR: {result.stderr[-300:]}")
else:
    print(result.stdout[-500:])

# Phase 4: Entanglement analysis
print(f"\n--- Phase 4: Entanglement analysis ---")
cmd = [PYTHON, "-c", f"""
import json, numpy as np, os, sys
sys.path.insert(0, '.')
from src.extraction.extract_persona_vectors_v2 import load_persona_vectors

model_short = '{MODEL_SHORT}'
pv_dir = f'results/persona_vectors/{{model_short}}'
if not os.path.exists(pv_dir):
    # Try with path-based name
    import glob
    dirs = glob.glob(f'results/persona_vectors/*Llama-2-7b*')
    if dirs:
        pv_dir = dirs[0]
        print(f'Found dir: {{pv_dir}}')

traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
vectors = {{}}
for t in traits:
    f = os.path.join(pv_dir, f'vector_{{t}}.npy')
    if os.path.exists(f):
        vectors[t] = np.load(f)

if len(vectors) >= 2:
    from itertools import combinations
    cosines = {{}}
    for t1, t2 in combinations(vectors.keys(), 2):
        v1, v2 = vectors[t1], vectors[t2]
        cos = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        cosines[f'{{t1}}-{{t2}}'] = round(float(cos), 4)
    
    result = {{
        'model': model_short,
        'n_traits': len(vectors),
        'cosine_similarities': cosines,
        'mean_abs_cos': round(float(np.mean(list(cosines.values()))), 4),
    }}
    outf = f'results/entanglement_metrics/{{model_short}}_entanglement.json'
    os.makedirs(os.path.dirname(outf), exist_ok=True)
    with open(outf, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
else:
    print(f'Only {{len(vectors)}} traits available, need >= 2')
"""]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
if result.returncode != 0:
    print(f"  ERROR: {result.stderr[-300:]}")
else:
    print(result.stdout)

print(f"\n{'='*60}")
print("INSTRUCTION-TUNING ABLATION COMPLETE")
print(f"{'='*60}")
