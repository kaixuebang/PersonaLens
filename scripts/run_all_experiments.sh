#!/bin/bash
# Master experiment runner for PersonaLens
# Manages GPU allocation and runs experiments in parallel
#
# Usage:
#   bash scripts/run_all_experiments.sh              # Run everything
#   bash scripts/run_all_experiments.sh --new-only    # Only new models
#   bash scripts/run_all_experiments.sh --ablations   # Only ablation experiments
#
# GPU Strategy:
#   GPU 0 (24GB free)  → Qwen2.5-7B-Instruct
#   GPU 3 (19GB free)  → Mistral-7B-Instruct-v0.1
#   GPU 5 (10GB free)  → Qwen2.5-1.5B-Instruct + small model ablations
#   GPU 7 (10GB free)  → Gemma-2 localization + small model ablations
#
# HuggingFace mirror for China:
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate personaforge_env

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()  { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
err() { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

MODE="${1:-all}"

###############################################################################
# Phase 1: Collect activations + Extract vectors + Steering
# This is the heaviest step for new models (requires model loading)
###############################################################################

run_activation_pipeline() {
    local model="$1"
    local device="cuda:$2"
    local log_file="$REPO_ROOT/results/logs/${model//\//_}_pipeline.log"
    mkdir -p "$REPO_ROOT/results/logs"
    
    log "Starting pipeline for $model on $device"
    
    conda run --no-banner -n personaforge_env python -u "$REPO_ROOT/scripts/run_pipeline.py" \
        --model "$model" \
        --trait big5 \
        --device "$device" \
        --skip_localize \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        ok "Pipeline complete for $model"
    else
        err "Pipeline FAILED for $model"
        return 1
    fi
}

run_localization() {
    local model="$1"
    local device="cuda:$2"
    local traits="$3"  # comma-separated or "big5"
    local log_file="$REPO_ROOT/results/logs/${model//\//_}_localization.log"
    mkdir -p "$REPO_ROOT/results/logs"
    
    log "Starting localization for $model on $device (traits: $traits)"
    
    if [ "$traits" = "big5" ]; then
        for trait in openness conscientiousness extraversion agreeableness neuroticism; do
            # Check if already done
            model_short="${model//\//_}"
            result_file="$REPO_ROOT/results/localization/${model_short}/refined_${trait}.json"
            if [ -f "$result_file" ]; then
                warn "Skipping $trait for $model - already exists"
                continue
            fi
            
            conda run --no-banner -n personaforge_env python -u \
                "$REPO_ROOT/src/localization/localize_circuits_v2.py" \
                --model "$model" \
                --trait "$trait" \
                --device "$device" \
                --n_samples 10 \
                2>&1 | tee -a "$log_file"
        done
    else
        for trait in $(echo "$traits" | tr ',' ' '); do
            model_short="${model//\//_}"
            result_file="$REPO_ROOT/results/localization/${model_short}/refined_${trait}.json"
            if [ -f "$result_file" ]; then
                warn "Skipping $trait for $model - already exists"
                continue
            fi
            
            conda run --no-banner -n personaforge_env python -u \
                "$REPO_ROOT/src/localization/localize_circuits_v2.py" \
                --model "$model" \
                --trait "$trait" \
                --device "$device" \
                --n_samples 10 \
                2>&1 | tee -a "$log_file"
        done
    fi
    
    ok "Localization complete for $model"
}

run_bfi_v2() {
    local model="$1"
    local device="cuda:$2"
    local log_file="$REPO_ROOT/results/logs/${model//\//_}_bfi_v2.log"
    mkdir -p "$REPO_ROOT/results/logs"
    
    log "Starting BFI V2 behavioral evaluation for $model on $device"
    
    for trait in openness conscientiousness extraversion agreeableness neuroticism; do
        # Check if already done
        model_short="${model//\//_}"
        result_file="$REPO_ROOT/results/bfi_behavioral_v2/${model_short}/responses_${trait}.json"
        if [ -f "$result_file" ]; then
            warn "Skipping BFI V2 $trait for $model - already exists"
            continue
        fi
        
        conda run --no-banner -n personaforge_env python -u \
            "$REPO_ROOT/src/evaluation/eval_bfi_behavioral_v2.py" \
            --model "$model" \
            --trait "$trait" \
            --device "$device" \
            2>&1 | tee -a "$log_file"
    done
    
    ok "BFI V2 complete for $model"
}

###############################################################################
# Phase 2: Ablation experiments (lighter, can run on any GPU)
###############################################################################

run_ood() {
    local model="$1"
    local model_short="${model//\//_}"
    local activations_dir="$REPO_ROOT/results/activations/$model_short"
    local output_dir="$REPO_ROOT/results/ood_results"
    
    if [ ! -d "$activations_dir" ]; then
        err "No activations for $model, skipping OOD"
        return 1
    fi
    
    log "Running OOD generalization for $model"
    conda run --no-banner -n personaforge_env python -u \
        "$REPO_ROOT/src/evaluation/eval_ood_generalization.py" \
        --activations_dir "$activations_dir" \
        --trait all \
        --output_dir "$output_dir"
    ok "OOD complete for $model"
}

run_shuffle() {
    local model="$1"
    local device="cuda:$2"
    
    log "Running shuffle-label baseline for $model on $device"
    conda run --no-banner -n personaforge_env python -u \
        "$REPO_ROOT/src/evaluation/eval_shuffle_label_baseline.py" \
        --model "$model" \
        --n_permutations 100 \
        --device "$device"
    ok "Shuffle baseline complete for $model"
}

run_null_ortho() {
    local model="$1"
    local device="cuda:$2"
    
    log "Running null orthogonality for $model on $device"
    conda run --no-banner -n personaforge_env python -u \
        "$REPO_ROOT/src/evaluation/eval_null_orthogonality.py" \
        --model "$model" \
        --device "$device"
    ok "Null orthogonality complete for $model"
}

###############################################################################
# Execution
###############################################################################

mkdir -p "$REPO_ROOT/results/logs"

if [ "$MODE" = "--new-only" ] || [ "$MODE" = "all" ]; then
    log "========================================="
    log "Phase 1: New Model Pipelines"
    log "========================================="
    
    # Launch all 3 new models in parallel on different GPUs
    # Each runs: activations → vectors → steering
    
    log "Launching Qwen2.5-7B on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 run_activation_pipeline "Qwen/Qwen2.5-7B-Instruct" 0 &
    PID_QWEN7B=$!
    
    log "Launching Mistral-7B on GPU 3..."
    CUDA_VISIBLE_DEVICES=3 run_activation_pipeline "mistralai/Mistral-7B-Instruct-v0.1" 0 &
    PID_MISTRAL=$!
    
    log "Launching Qwen2.5-1.5B on GPU 5..."
    CUDA_VISIBLE_DEVICES=5 run_activation_pipeline "Qwen/Qwen2.5-1.5B-Instruct" 0 &
    PID_QWEN15=$!
    
    # Wait for all pipelines
    log "Waiting for new model pipelines to finish..."
    wait $PID_QWEN7B $PID_MISTRAL $PID_QWEN15 2>/dev/null || true
    
    ok "All new model pipelines complete (or attempted)"
fi

if [ "$MODE" = "--ablations" ] || [ "$MODE" = "all" ]; then
    log "========================================="
    log "Phase 2: Ablation Experiments"
    log "========================================="
    
    # OOD generalization (CPU-only, no model needed)
    log "Running OOD generalization for existing models..."
    for model in Qwen/Qwen3-0.6B TinyLlama/TinyLlama-1.1B-Chat-v1.0 unsloth/gemma-2-2b-it unsloth/Llama-3.2-1B-Instruct; do
        model_short="${model//\//_}"
        if [ -d "$REPO_ROOT/results/ood_results/$model_short" ]; then
            warn "OOD already exists for $model, skipping"
            continue
        fi
        run_ood "$model" &
    done
    wait
    
    # Shuffle label baseline (needs model loading, run on free GPUs)
    log "Running shuffle label baselines..."
    for model in Qwen/Qwen3-0.6B TinyLlama/TinyLlama-1.1B-Chat-v1.0 unsloth/gemma-2-2b-it unsloth/Llama-3.2-1B-Instruct; do
        model_short="${model//\//_}"
        if [ -d "$REPO_ROOT/results/shuffle_label_baseline_results/$model_short" ]; then
            warn "Shuffle baseline already exists for $model, skipping"
            continue
        fi
        run_shuffle "$model" 0 &
    done
    wait
    
    # Null orthogonality (needs model loading)
    log "Running null orthogonality baselines..."
    for model in Qwen/Qwen3-0.6B TinyLlama/TinyLlama-1.1B-Chat-v1.0 unsloth/gemma-2-2b-it unsloth/Llama-3.2-1B-Instruct; do
        model_short="${model//\//_}"
        if [ -d "$REPO_ROOT/results/null_orthogonality_results/$model_short" ]; then
            warn "Null orthogonality already exists for $model, skipping"
            continue
        fi
        run_null_ortho "$model" 0 &
    done
    wait
fi

if [ "$MODE" = "--gemma-localization" ] || [ "$MODE" = "all" ]; then
    log "========================================="
    log "Phase 2.5: Gemma-2 Missing Localization"
    log "========================================="
    
    # Gemma-2 only has openness, needs 4 more traits
    for trait in agreeableness conscientiousness extraversion neuroticism; do
        result_file="$REPO_ROOT/results/localization/unsloth_gemma-2-2b-it/refined_${trait}.json"
        if [ -f "$result_file" ]; then
            warn "Skipping Gemma-2 $trait - already exists"
            continue
        fi
        log "Running Gemma-2 localization for $trait..."
        CUDA_VISIBLE_DEVICES=5 run_localization "unsloth/gemma-2-2b-it" 0 "$trait"
    done
fi

if [ "$MODE" = "--localization" ] || [ "$MODE" = "all" ]; then
    log "========================================="
    log "Phase 3: Localization for new models"
    log "========================================="
    
    # These need model loading on specific GPUs
    # Run sequentially since each uses significant GPU memory
    for model in Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.1 Qwen/Qwen2.5-1.5B-Instruct; do
        model_short="${model//\//_}"
        # Check if localization already done
        result_file="$REPO_ROOT/results/localization/${model_short}/refined_openness.json"
        if [ -f "$result_file" ]; then
            warn "Localization already exists for $model, skipping"
            continue
        fi
        
        log "Running localization for $model..."
        CUDA_VISIBLE_DEVICES=0 run_localization "$model" 0 "big5"
    done
fi

if [ "$MODE" = "--bfi-v2" ] || [ "$MODE" = "all" ]; then
    log "========================================="
    log "Phase 4: BFI V2 Behavioral Evaluation"
    log "========================================="
    
    for model in Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.1 Qwen/Qwen2.5-1.5B-Instruct; do
        model_short="${model//\//_}"
        result_file="$REPO_ROOT/results/bfi_behavioral_v2/${model_short}/responses_openness.json"
        if [ -f "$result_file" ]; then
            warn "BFI V2 already exists for $model, skipping"
            continue
        fi
        
        log "Running BFI V2 for $model..."
        CUDA_VISIBLE_DEVICES=0 run_bfi_v2 "$model" 0
    done
fi

if [ "$MODE" = "--new-ablations" ] || [ "$MODE" = "all" ]; then
    log "========================================="
    log "Phase 5: Ablations for new models"
    log "========================================="
    
    for model in Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.1 Qwen/Qwen2.5-1.5B-Instruct; do
        model_short="${model//\//_}"
        activations_dir="$REPO_ROOT/results/activations/$model_short"
        
        if [ ! -d "$activations_dir" ]; then
            warn "No activations for $model, skipping ablations"
            continue
        fi
        
        # OOD
        if [ ! -d "$REPO_ROOT/results/ood_results/$model_short" ]; then
            run_ood "$model" &
        fi
        
        # Shuffle + Null need model loading, run sequentially
        if [ ! -d "$REPO_ROOT/results/shuffle_label_baseline_results/$model_short" ]; then
            CUDA_VISIBLE_DEVICES=0 run_shuffle "$model" 0
        fi
        if [ ! -d "$REPO_ROOT/results/null_orthogonality_results/$model_short" ]; then
            CUDA_VISIBLE_DEVICES=0 run_null_ortho "$model" 0
        fi
    done
    wait
fi

log "========================================="
ok "ALL EXPERIMENTS COMPLETE"
log "========================================="
log "Results summary:"
for dir in results/activations results/persona_vectors results/localization results/steering_results results/bfi_behavioral_v2 results/ood_results results/shuffle_label_baseline_results results/null_orthogonality_results; do
    if [ -d "$REPO_ROOT/$dir" ]; then
        count=$(ls "$REPO_ROOT/$dir" 2>/dev/null | wc -l)
        log "  $dir: $count entries"
    fi
done
