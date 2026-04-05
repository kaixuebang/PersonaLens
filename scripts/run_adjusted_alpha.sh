#!/bin/bash
# Run BFI behavioral eval with adjusted alphas to normalize perturbation ratios
# across models. Target: ~100% perturbation ratio (alpha/||h|| ≈ 1.0)
#
# Original perturbation ratios at alpha=6:
#   Q2.5-7B:   19.7% → need alpha≈30  (5x)
#   Q2.5-1.5:  18.4% → need alpha≈33  (5.5x)
#   Q3-0.6:    10.8% → need alpha≈56  (9x)
#   Gemma-2B:   4.3% → need alpha≈140 (23x)
#
# We use 10x for all models (alpha max = 60) as the primary experiment,
# and additionally 40x for Gemma-2B (alpha max = 240).

PYTHON=/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python
REPO=/data1/tongjizhou/persona
OUTPUT_DIR=$REPO/results/bfi_adjusted_alpha

mkdir -p $OUTPUT_DIR

run_model() {
    local GPU=$1
    local MODEL=$2
    local ALPHAS=$3
    local LABEL=$4

    echo "[$(date)] Starting $MODEL ($LABEL) on GPU $GPU with alphas=$ALPHAS"

    HF_ENDPOINT=https://hf-mirror.com \
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONPATH=$REPO \
    nohup $PYTHON -u -m src.evaluation.eval_bfi_behavioral_v2 \
        --model "$MODEL" \
        --trait all \
        --output_dir "$OUTPUT_DIR" \
        --alphas "$ALPHAS" \
        > "$OUTPUT_DIR/${LABEL}.log" 2>&1 &

    echo "[$(date)] PID=$! for $LABEL on GPU $GPU"
}

# ============================================================
# Phase 1: BFI response generation with adjusted alphas
# ============================================================

# Qwen2.5-7B: 5x scaling → perturbation ~99%
# Needs ~14GB VRAM
run_model 0 "Qwen/Qwen2.5-7B-Instruct" \
    "-30,-20,-10,0,10,20,30" \
    "Q2.5-7_adj5x" &

# Qwen2.5-1.5B: 5x scaling → perturbation ~92%
# Needs ~4GB VRAM
run_model 1 "Qwen/Qwen2.5-1.5B-Instruct" \
    "-30,-20,-10,0,10,20,30" \
    "Q2.5-1.5_adj5x" &

# Qwen3-0.6B: 9x scaling → perturbation ~97%
# Needs ~2GB VRAM
run_model 2 "Qwen/Qwen3-0.6B" \
    "-54,-36,-18,0,18,36,54" \
    "Q3-0.6_adj9x" &

# Gemma-2B: 23x scaling → perturbation ~100%
# Needs ~6GB VRAM
run_model 3 "unsloth/gemma-2-2b-it" \
    "-138,-92,-46,0,46,92,138" \
    "Gemma-2B_adj23x" &

wait
echo ""
echo "[$(date)] All BFI response generation complete!"

# ============================================================
# Phase 2: LLM-as-Judge scoring
# ============================================================
echo "[$(date)] Starting LLM-as-Judge scoring..."

for model_dir in $OUTPUT_DIR/*/; do
    model_name=$(basename "$model_dir")
    echo "  Scoring $model_name..."

    HF_ENDPOINT=https://hf-mirror.com \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=$REPO \
    $PYTHON -u -m src.evaluation.judge_bfi_behavioral \
        --judge unsloth/gemma-2-2b-it \
        --model-dir "$model_dir" \
        >> "$OUTPUT_DIR/judge_${model_name}.log" 2>&1
done

echo "[$(date)] All judge scoring complete!"
echo "Results in: $OUTPUT_DIR"
