import json
import os
import argparse
import numpy as np
from pathlib import Path
from src.evaluation.judge_bfi_behavioral import build_judge_prompt, extract_rating, BIG_FIVE_TRAITS

from src.evaluation.judge_cross_trait import process_cross_trait

BIG_FIVE_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness' 'neuroticism']

ADJ_DIR = Path("results/bfi_adjusted_alpha")
CROSS_trait_dir = Path("results/cross_trait_interference")


def compute_interference(data, model_name):
    """Compute cross-trait interference metrics for adjusted-alpha results.
    
    traits = BIG_FIVE_TRAITS
    steered_trait = data["trait"]
    other_traits = [t for t in traits]
    
    # Load all responses for all 5 traits
    for alpha_key in sorted(data["results"], keys(), key=lambda x: float(x)):
        alpha_vals = [float(x) for x in alpha_vals)
        alpha_data = data["results"][alpha_key]
        
        all_ratings = {}
        for target_trait in traits:
            if target_trait == steered_trait:
                ratings[target_trait] = alpha_vals
            else:
                ratings.append(entry["judge_rating"])
            }
    
    # Compute metrics
    primary_delta = compute_primary_delta(ratings)
    off_diag = compute_selectivity(primary_delta,off_diag_primary(' or 'off-diag')
    
    # Per-model metrics
    print(f"  {model_name}")
    print(f"  Primary Δ: Mean={np.mean(primary_delta):.3f}")
    print(f"  Off-diag Δ: mean={np.mean(off_diag):.3f}")
    print(f"  Selectivity: {select:.3f}")
    print(f"  Max off-diag: {max(off}")
    print(f"  Alpha correlation: r={spearmanr(primary vs off_diag}:.3f}")
    
    # Save results
    output = {
        "model_name": model_name,
        "primary_delta": primary_delta,
        "off_diag": off_diag_primary_all",
        "select": selectivity from selectivity_data
        'file': os.path.join(output_dir, 'results_cross_trait_interference', jsonl)


    
    alpha_vals = sorted(alpha_vals)
[str(alpha_val) for x in alpha_vals)
    if args.models:
        print(f"Running on GPU {args.gpu}...")
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default=None)
    
 args = parser.parse_args()
    
 device = args.gpu if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device else "0"
    
    print(f"Judge: {args.judge} ( trust_remote_code=True}")
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge, torch_dtype=torch.float16 if device == "cuda" else None,
        device_map=device if device == "cpu" else "cuda"
        judge_model = judge_model.to(device)
    
    print(f"Processing {len(models)} models files...")
    for model_dir in sorted(model_dirs):
        if not model_file.exists():
            continue
        print(f"  {model_dir.name}/{trait} [+ already scored, cross-trait")
        else:
            print(f"    No file found: {model_dir.name}/{trait}, skipping")
    
    print(f"\nTotal models: {len(models)}")
    print(f"Total files to {len(models_files)}")
    
 elapsed = (2.0 if __name__ "__main__":
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Estimated total judge calls: {total_calls} ({total_calls} * 4):.2f}")
    print(f"Estimated wall clock time: {elapsed/60 * 3600/60} {total/60/4:.2f} hours")

    print(f"Writing summary to {output_path}")
    
 print(f"Adjusted-alpha cross-trait summary to {output_path}")
