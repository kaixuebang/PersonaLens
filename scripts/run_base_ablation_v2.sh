#!/bin/bash
set -e
cd /data1/tongjizhou/persona

export CUDA_VISIBLE_DEVICES=${1:-4}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python
BASE=/data0/shizitong/models/Llama-2-7b-hf
MODEL_SHORT=_data0_shizitong_models_Llama-2-7b-hf
ACT_DIR=activations/${MODEL_SHORT}
ACT_OUTER=${ACT_DIR}

echo "============================================================"
echo "INSTRUCTION-TUNING ABLATION: Llama-2-7b-hf (BASE)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Phase 1: Collect activations
echo ""
echo "--- Phase 1: Collect activations ---"
for TRAIT in openness conscientiousness extraversion agreeableness neuroticism; do
    echo "  Collecting $TRAIT..."
    $PYTHON -m src.localization.collect_activations \
        --model $BASE --trait $TRAIT --device cuda \
        --output_dir $ACT_DIR 2>&1 | grep -E "Saved|Error|Traceback" || true
done

# Phase 2: Extract persona vectors (point to the inner model_short dir)
echo ""
echo "--- Phase 2: Extract persona vectors ---"
$PYTHON -m src.extraction.extract_persona_vectors_v2 \
    --activations_dir ${ACT_DIR}/${MODEL_SHORT} \
    --output_dir results/persona_vectors/${MODEL_SHORT} \
    --trait all 2>&1 | tail -10

# Phase 3: Steering eval (uses the model directly)
echo ""
echo "--- Phase 3: Steering evaluation ---"
$PYTHON -m src.evaluation.eval_bfi_behavioral_v2 \
    --model $BASE --trait all --device cuda \
    --output_dir results/bfi_behavioral_v2 2>&1 | tail -10

# Phase 4: Judge
echo ""
echo "--- Phase 4: Judging ---"
$PYTHON -m src.evaluation.judge_bfi_behavioral \
    --model _data0_shizitong_models_Llama-2-7b-hf \
    --trait all --device cuda 2>&1 | tail -10

echo ""
echo "============================================================"
echo "ABLATION COMPLETE"
echo "============================================================"
