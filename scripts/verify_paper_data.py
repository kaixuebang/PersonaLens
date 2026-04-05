"""
Verify all paper table data against source JSON files.
Computes: steering deltas, monotonicity, per-trait breakdowns.
"""
import json
import os
import numpy as np
from pathlib import Path
from scipy import stats

REPO = Path("/data1/tongjizhou/persona")
BFI_DIR = REPO / "results" / "bfi_behavioral_v2"
ADJ_DIR = REPO / "results" / "bfi_adjusted_alpha"
VECTORS_DIR = REPO / "results" / "persona_vectors"

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def extract_judge_scores_from_json(json_path):
    """Extract judge scores from a responses JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    # The structure varies - need to find judge scores
    scores_by_alpha = {}
    
    if isinstance(data, list):
        for entry in data:
            alpha = entry.get("alpha", entry.get("steering_alpha", None))
            if alpha is None:
                continue
            # Look for judge scores in various fields
            judge_score = entry.get("judge_score", entry.get("bfi_score", None))
            if judge_score is not None:
                if alpha not in scores_by_alpha:
                    scores_by_alpha[alpha] = []
                scores_by_alpha[alpha].append(float(judge_score))
    elif isinstance(data, dict):
        # Check for alpha-keyed structure
        if "results" in data:
            for entry in data["results"]:
                alpha = entry.get("alpha", entry.get("steering_alpha", None))
                judge_score = entry.get("judge_score", entry.get("bfi_score", None))
                if alpha is not None and judge_score is not None:
                    if alpha not in scores_by_alpha:
                        scores_by_alpha[alpha] = []
                    scores_by_alpha[alpha].append(float(judge_score))
        else:
            # Maybe responses list at top level
            for key in ["responses", "items", "data"]:
                if key in data and isinstance(data[key], list):
                    for entry in data[key]:
                        alpha = entry.get("alpha", entry.get("steering_alpha", None))
                        judge_score = entry.get("judge_score", entry.get("bfi_score", None))
                        if alpha is not None and judge_score is not None:
                            if alpha not in scores_by_alpha:
                                scores_by_alpha[alpha] = []
                            scores_by_alpha[alpha].append(float(judge_score))
    
    return scores_by_alpha

def compute_steering_metrics(scores_by_alpha):
    """Compute delta and monotonicity from alpha->scores mapping."""
    if not scores_by_alpha:
        return None, None
    
    alphas = sorted(scores_by_alpha.keys())
    means = [np.mean(scores_by_alpha[a]) for a in alphas]
    
    delta = max(means) - min(means)
    
    # Spearman correlation
    if len(alphas) >= 3:
        r, p = stats.spearmanr(alphas, means)
    else:
        r = None
    
    return delta, r

def inspect_json_structure(json_path):
    """Print structure of a JSON file for debugging."""
    with open(json_path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"  List of {len(data)} items")
        if data:
            print(f"  First item keys: {list(data[0].keys())[:10]}")
            # Show first item sample
            sample = {k: (v if not isinstance(v, (list, dict)) else f"<{type(v).__name__}>") for k, v in data[0].items()}
            print(f"  Sample: {json.dumps(sample, indent=2)[:500]}")
    elif isinstance(data, dict):
        print(f"  Dict with keys: {list(data.keys())[:15]}")
        for k in list(data.keys())[:3]:
            v = data[k]
            if isinstance(v, list):
                print(f"  '{k}': list of {len(v)} items")
                if v:
                    print(f"    First item keys: {list(v[0].keys())[:10] if isinstance(v[0], dict) else 'N/A'}")
            elif isinstance(v, dict):
                print(f"  '{k}': dict with {len(v)} keys")
            else:
                print(f"  '{k}': {type(v).__name__} = {str(v)[:100]}")
    return data

# First, inspect the structure
print("=" * 80)
print("INSPECTING JSON STRUCTURES")
print("=" * 80)

# Check original BFI
print("\n--- Original BFI (Q2.5-1.5B openness) ---")
orig_path = BFI_DIR / "Qwen_Qwen2.5-1.5B-Instruct" / "responses_openness.json"
orig_data = inspect_json_structure(orig_path)

print("\n--- Adjusted BFI (Q2.5-1.5B openness) ---")
adj_path = ADJ_DIR / "Qwen_Qwen2.5-1.5B-Instruct" / "responses_openness.json"
adj_data = inspect_json_structure(adj_path)

print("\n--- Vectors analysis (Q2.5-1.5B openness) ---")
vec_path = VECTORS_DIR / "Qwen_Qwen2.5-1.5B-Instruct" / "openness" / "analysis_v2_openness.json"
vec_data = inspect_json_structure(vec_path)
