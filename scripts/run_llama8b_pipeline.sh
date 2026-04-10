#!/bin/bash
# LLaMA-3.1-8B pipeline - step by step
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,7
PYTHON=/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python
MODEL=unsloth/Llama-3.1-8B-Instruct
LOGDIR=results/llama8b_pipeline
mkdir -p $LOGDIR

# Step 1: Collect activations
echo "$(date): Step 1 - Collecting activations..."
PYTHONPATH=/data1/tongjizhou/persona $PYTHON -u src/localization/collect_activations.py \
    --model $MODEL --traits openness,conscientiousness,extraversion,agreeableness,neuroticism \
    --device auto --output_dir activations/ \
    > $LOGDIR/01_activations.log 2>&1
echo "$(date): Step 1 done. Exit code: $?"

# Step 2: Extract vectors
echo "$(date): Step 2 - Extracting vectors..."
PYTHONPATH=/data1/tongjizhou/persona $PYTHON -u src/extraction/extract_persona_vectors_v2.py \
    --model $MODEL --traits openness,conscientiousness,extraversion,agreeableness,neuroticism \
    --output_dir persona_vectors/ \
    > $LOGDIR/02_extraction.log 2>&1
echo "$(date): Step 2 done. Exit code: $?"

# Step 3: Steering + BFI eval
echo "$(date): Step 3 - Steering..."
PYTHONPATH=/data1/tongjizhou/persona $PYTHON -u src/evaluation/judge_bfi_behavioral.py \
    --model $MODEL --traits openness,conscientiousness,extraversion,agreeableness,neuroticism \
    --output_dir results/bfi_behavioral_v2/ \
    > $LOGDIR/03_steering.log 2>&1
echo "$(date): Step 3 done. Exit code: $?"

# Step 4: Entanglement
echo "$(date): Step 4 - Entanglement metrics..."
PYTHONPATH=/data1/tongjizhou/persona $PYTHON -u scripts/compute_entanglement_metrics.py \
    --model $MODEL \
    --output_dir results/entanglement_metrics/ \
    > $LOGDIR/04_entanglement.log 2>&1
echo "$(date): Step 4 done. Exit code: $?"

echo "$(date): Pipeline complete!"
