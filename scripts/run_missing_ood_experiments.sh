#!/bin/bash
# Run missing OOD experiments for Qwen3, Qwen2.5, and TinyLlama
# All persona vectors already exist, so this only runs OOD evaluation

set -e

MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

TRAITS=(
    "conscientiousness"
    "extraversion"
    "agreeableness"
    "neuroticism"
)

echo "========================================================================"
echo "Running Missing OOD Experiments"
echo "========================================================================"
echo ""
echo "This will complete the OOD generalization table (Table 4) in the paper."
echo "Estimated time: ~2-3 hours GPU time"
echo ""

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Model: $model"
    echo "----------------------------------------"
    
    for trait in "${TRAITS[@]}"; do
        echo "  Running OOD evaluation for $trait..."
        
        python src/evaluation/eval_ood_generalization.py \
            --model "$model" \
            --trait "$trait" \
            --device cuda \
            || echo "    ⚠ Failed for $model - $trait"
        
        echo "  ✓ Completed $trait"
    done
    
    echo ""
done

echo "========================================================================"
echo "Aggregating Results"
echo "========================================================================"

python scripts/aggregate_ood_data.py

echo ""
echo "✓ All OOD experiments complete!"
echo ""
echo "Next steps:"
echo "  1. Check ood_aggregated_results.json for updated data"
echo "  2. Update Table 4 in paper/main.tex with new values"
echo "  3. Verify all values are >0.90 (strong OOD generalization)"
