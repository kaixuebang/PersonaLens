#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2
PYTHON=/data1/tongjizhou/miniconda3/envs/personaforge_env/bin/python
REPO=/data1/tongjizhou/persona

cd $REPO
echo "Starting adjusted-alpha cross-trait judging at $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

$PYTHON -u src/evaluation/judge_cross_trait.py \
    --judge unsloth/gemma-2-2b-it \
    --source adjusted \
    --device cuda 2>&1

echo "Finished at $(date)"
