#!/bin/bash
# Full pipeline for a single model: collect activations -> extract vectors -> steer -> judge
# Usage: bash run_new_model_pipeline.sh <model_path> <gpu_ids> <model_short_name>
# Example: bash run_new_model_pipeline.sh /data0/shizitong/models/Phi3-mini-128k-instruct 4 phi3-mini

set -e

MODEL_PATH=$1
GPU_IDS=$2
MODEL_SHORT=$3
REPO=/data1/tongjizhou/persona
PYTHON=/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python

# Compute the directory-safe model name (same as what the scripts use internally)
MODEL_DIR_NAME=$(echo "$MODEL_PATH" | sed 's|/$||' | tr '/' '_')

echo "=== Pipeline for $MODEL_SHORT ==="
echo "Model path: $MODEL_PATH"
echo "Dir name: $MODEL_DIR_NAME"
echo "GPUs: $GPU_IDS"
echo "Start: $(date)"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTHONPATH=$REPO

# Step 1: Collect activations for all 5 traits
echo ""
echo "=== Step 1: Collecting activations ==="
for TRAIT in openness conscientiousness extraversion agreeableness neuroticism; do
    echo "--- $TRAIT at $(date) ---"
    $PYTHON -u -m src.localization.collect_activations \
        --model "$MODEL_PATH" \
        --trait $TRAIT \
        --output_dir $REPO/results/activations \
        --device auto \
        2>&1 || echo "WARNING: $TRAIT failed"
done

# Step 2: Extract persona vectors
echo ""
echo "=== Step 2: Extracting persona vectors ==="
ACTDIR=$REPO/results/activations/$MODEL_DIR_NAME
for TRAIT in openness conscientiousness extraversion agreeableness neuroticism; do
    echo "--- $TRAIT ---"
    $PYTHON -u -m src.extraction.extract_persona_vectors_v2 \
        --activations_dir "$ACTDIR" \
        --trait $TRAIT \
        --output_dir $REPO/results/persona_vectors/$MODEL_DIR_NAME \
        2>&1 || echo "WARNING: $TRAIT extraction failed"
done

# Step 3: Determine best layer (use middle layer for safety)
NUM_LAYERS=$(cat "$MODEL_PATH/config.json" | python3 -c "import json,sys; print(json.load(sys.stdin)['num_hidden_layers'])")
MID_LAYER=$((NUM_LAYERS / 2))
echo ""
echo "=== Using middle layer: $MID_LAYER (of $NUM_LAYERS) ==="

# Step 4: Behavioral steering evaluation
echo ""
echo "=== Step 3: Behavioral steering eval ==="
$PYTHON -u -m src.evaluation.eval_bfi_behavioral_v2 \
    --model "$MODEL_PATH" \
    --trait all \
    --output_dir $REPO/results/bfi_behavioral_v2 \
    --device auto \
    --layers $MID_LAYER \
    2>&1 || echo "WARNING: steering eval failed"

# Step 5: Judge scoring
echo ""
echo "=== Step 4: Judge scoring ==="
$PYTHON -u -c "
import json, os, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, '$REPO')
os.chdir('$REPO')
from src.evaluation.judge_bfi_behavioral import process_file

JUDGE_MODEL = 'unsloth/gemma-2-2b-it'
MODEL_DIR_NAME = '$MODEL_DIR_NAME'
DATA_DIR = f'results/bfi_behavioral_v2/{MODEL_DIR_NAME}'
TRAITS = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']

print(f'Loading judge for {MODEL_DIR_NAME}...')
tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.float16, device_map='cuda', trust_remote_code=True, local_files_only=True)
model.eval()

for trait in TRAITS:
    fpath = os.path.join(DATA_DIR, f'responses_{trait}.json')
    if os.path.exists(fpath):
        print(f'Judging {trait}...')
        process_file(fpath, model, tokenizer, trait, 'cuda')
    else:
        print(f'SKIP {trait}')

del model, tokenizer
torch.cuda.empty_cache()
print('Judge done.')
" 2>&1 || echo "WARNING: judge failed"

echo ""
echo "=== Pipeline complete for $MODEL_SHORT at $(date) ==="
