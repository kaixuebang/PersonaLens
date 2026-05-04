#!/bin/bash
# Full GPU pipeline for Mistral-7B-Instruct-v0.2
# All model inference on single GPU, no CPU offloading
# Usage: bash run_mistral_v02_gpu.sh <gpu_id>
set -e

GPU_ID=${1:-3}
MODEL_PATH="/data0/shizitong/models/Mistral-7B-Instruct-v0.2"
MODEL_SHORT="mistralai_Mistral-7B-Instruct-v0.2"
REPO="/data1/tongjizhou/persona"
PYTHON="/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python"
NUM_LAYERS=32
MID_LAYER=16

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=$REPO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Mistral-7B-v0.2 Full GPU Pipeline ==="
echo "GPU: $GPU_ID | Model: $MODEL_PATH"
echo "Start: $(date)"

cd $REPO

# ============================================================
# PHASE 1: Activation Collection (GPU)
# ============================================================
echo ""
echo "=== PHASE 1: Activation Collection ==="
for TRAIT in openness conscientiousness extraversion agreeableness neuroticism; do
    echo "--- $TRAIT at $(date) ---"
    $PYTHON -u -m src.localization.collect_activations \
        --model "$MODEL_PATH" \
        --trait $TRAIT \
        --output_dir $REPO/results/activations \
        --device cuda \
        2>&1 || echo "WARNING: $TRAIT activation collection issue"
done

# ============================================================
# PHASE 2: Vector Extraction (CPU but fast - numpy/sklearn)
# ============================================================
echo ""
echo "=== PHASE 2: Vector Extraction ==="
ACTDIR="$REPO/results/activations/_data0_shizitong_models_Mistral-7B-Instruct-v0.2"
for TRAIT in openness conscientiousness extraversion agreeableness neuroticism; do
    echo "--- $TRAIT ---"
    $PYTHON -u -m src.extraction.extract_persona_vectors_v2 \
        --activations_dir "$ACTDIR" \
        --trait $TRAIT \
        --output_dir "$REPO/results/persona_vectors/_data0_shizitong_models_Mistral-7B-Instruct-v0.2" \
        2>&1 || echo "WARNING: $TRAIT extraction issue"
done

# ============================================================
# PHASE 3: Steering Evaluation (GPU — use mid layer)
# ============================================================
echo ""
echo "=== PHASE 3: Steering Evaluation (mid layer L${MID_LAYER}) ==="
$PYTHON -u -m src.evaluation.eval_bfi_behavioral_v2 \
    --model "$MODEL_PATH" \
    --trait all \
    --output_dir $REPO/results/bfi_behavioral_v2 \
    --device cuda \
    --layers $MID_LAYER \
    2>&1 || echo "WARNING: steering eval issue"

# ============================================================
# PHASE 4: Judge Scoring (GPU — load judge model)
# ============================================================
echo ""
echo "=== PHASE 4: Judge Scoring ==="
$PYTHON -u -c "
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '$REPO')
os.chdir('$REPO')
from src.evaluation.judge_bfi_behavioral import process_file

JUDGE_MODEL = 'unsloth/gemma-2-2b-it'
MODEL_DIR = '_data0_shizitong_models_Mistral-7B-Instruct-v0.2'
DATA_DIR = f'results/bfi_behavioral_v2/{MODEL_DIR}'
TRAITS = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']

print('Loading judge model on cuda...')
tok = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True, local_files_only=True)
judge = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL, torch_dtype=torch.float16,
    device_map='cuda', trust_remote_code=True, local_files_only=True
)
judge.eval()

for trait in TRAITS:
    fpath = os.path.join(DATA_DIR, f'responses_{trait}.json')
    if os.path.exists(fpath):
        print(f'Judging {trait}...')
        process_file(fpath, judge, tok, trait, 'cuda')
    else:
        print(f'SKIP {trait} — no responses file')

del judge, tok
torch.cuda.empty_cache()
print('Judge complete.')
" 2>&1 || echo "WARNING: judge scoring issue"

echo ""
echo "=== Pipeline complete at $(date) ==="
