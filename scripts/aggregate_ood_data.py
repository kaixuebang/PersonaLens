"""
Aggregate OOD Generalization Data for Paper Table

This script reads all OOD stability JSON files and computes the best
intermediate layer cosine similarity for each model/trait combination.

Generates data for Table 4 (OOD Generalization) in the paper.
"""

import json
import os
from pathlib import Path
import numpy as np

# Models and traits
MODELS = {
    "Qwen3": "Qwen_Qwen3-0.6B",
    "Qwen2.5": "Qwen_Qwen2.5-0.5B-Instruct",
    "TinyLlama": "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
    "Llama-3.2": "unsloth_Llama-3.2-1B-Instruct",
    "Gemma-2": "unsloth_gemma-2-2b-it",
}

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def find_best_intermediate_layer(data, n_layers):
    """
    Find best layer in intermediate range (middle 50% of layers).

    Args:
        data: Dict with layer -> {"mean": float, "std": float}
        n_layers: Total number of layers

    Returns:
        Tuple of (best_layer, best_cosine)
    """
    # Intermediate range: skip bottom 25% and top 25%
    start_idx = n_layers // 4
    end_idx = 3 * n_layers // 4

    intermediate_layers = range(start_idx, end_idx)

    best_layer = None
    best_cosine = -1.0

    for layer in intermediate_layers:
        layer_str = str(layer)
        if layer_str in data:
            cosine = data[layer_str]["mean"]
            if cosine > best_cosine:
                best_cosine = cosine
                best_layer = layer

    return best_layer, best_cosine


def aggregate_ood_data(ood_dirs=None):
    """Aggregate all OOD data into a summary table.
    
    Searches multiple directories for OOD results, preferring ood_results_fixed
    but falling back to ood_results for data not in the fixed directory.
    """

    if ood_dirs is None:
        ood_dirs = ["ood_results_fixed", "ood_results"]
    elif isinstance(ood_dirs, str):
        ood_dirs = [ood_dirs]

    ood_paths = [Path(d) for d in ood_dirs]

    # Model layer counts (from paper)
    layer_counts = {
        "Qwen3": 28,
        "Qwen2.5": 24,
        "TinyLlama": 22,
        "Llama-3.2": 16,
        "Gemma-2": 26,
    }

    results = {}

    for model_name, model_dir in MODELS.items():
        results[model_name] = {}

        for trait in TRAITS:
            found = False
            for ood_path in ood_paths:
                json_path = ood_path / model_dir / trait / "ood_stability.json"

                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)

                    # Check if summary already exists
                    if "summary" in data and "best_layer_cosine" in data["summary"]:
                        best_cosine = data["summary"]["best_layer_cosine"]
                        best_layer = data["summary"]["best_layer"]
                    else:
                        # Compute from mean_diff_cosine data
                        if "mean_diff_cosine" in data:
                            n_layers = layer_counts[model_name]
                            best_layer, best_cosine = find_best_intermediate_layer(
                                data["mean_diff_cosine"], n_layers
                            )
                        else:
                            best_layer, best_cosine = None, None

                    results[model_name][trait] = {
                        "cosine": best_cosine,
                        "layer": best_layer,
                        "available": True,
                    }
                    found = True
                    break  # Use first found (prefer ood_results_fixed)

            if not found:
                results[model_name][trait] = {
                    "cosine": None,
                    "layer": None,
                    "available": False,
                }

    return results


def print_latex_table(results):
    """Print LaTeX table for paper."""

    print("\\begin{table}[t]")
    print(
        "\\caption{\\textbf{Cross-model OOD generalization.} Mean cosine similarity between persona vectors extracted from disjoint scenario subsets. Values $>0.90$ indicate that vectors capture genuine trait abstractions rather than scenario-specific content.}"
    )
    print("\\label{tab:ood}")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print(
        "& \\textbf{Qwen3} & \\textbf{Qwen2.5} & \\textbf{TinyLlama} & \\textbf{Llama-3.2} & \\textbf{Gemma-2} \\\\"
    )
    print("\\midrule")

    trait_names = {
        "openness": "Openness",
        "conscientiousness": "Conscientiousness",
        "extraversion": "Extraversion",
        "agreeableness": "Agreeableness",
        "neuroticism": "Neuroticism",
    }

    for trait in TRAITS:
        row = f"OOD $\\cos$ ({trait_names[trait]})"
        for model in ["Qwen3", "Qwen2.5", "TinyLlama", "Llama-3.2", "Gemma-2"]:
            if (
                results[model][trait]["available"]
                and results[model][trait]["cosine"] is not None
            ):
                cosine = results[model][trait]["cosine"]
                row += f" & {cosine:.3f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_summary_stats(results):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("OOD Generalization Summary Statistics")
    print("=" * 80)

    for model in ["Qwen3", "Qwen2.5", "TinyLlama", "Llama-3.2", "Gemma-2"]:
        print(f"\n{model}:")

        available_traits = []
        cosines = []

        for trait in TRAITS:
            if (
                results[model][trait]["available"]
                and results[model][trait]["cosine"] is not None
            ):
                available_traits.append(trait)
                cosines.append(results[model][trait]["cosine"])

        if cosines:
            print(f"  Available traits: {len(available_traits)}/5")
            print(f"  Mean cosine: {np.mean(cosines):.3f}")
            print(f"  Min cosine: {np.min(cosines):.3f}")
            print(f"  Max cosine: {np.max(cosines):.3f}")
            print(f"  Traits: {', '.join(available_traits)}")
        else:
            print(f"  No OOD data available")


def main():
    print("Aggregating OOD Generalization Data...")
    print("=" * 80)

    results = aggregate_ood_data()

    # Print summary stats
    print_summary_stats(results)

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table for Paper (Table 4)")
    print("=" * 80 + "\n")
    print_latex_table(results)

    # Save results
    output_file = "ood_aggregated_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Check completeness
    print("\n" + "=" * 80)
    print("Data Completeness Check")
    print("=" * 80)

    total_cells = len(MODELS) * len(TRAITS)
    available_cells = sum(
        1
        for model in results.values()
        for trait_data in model.values()
        if trait_data["available"] and trait_data["cosine"] is not None
    )

    print(
        f"Available: {available_cells}/{total_cells} ({100 * available_cells / total_cells:.1f}%)"
    )

    if available_cells < total_cells:
        print("\n⚠ Missing data for:")
        for model_name, model_data in results.items():
            missing = [
                trait
                for trait, data in model_data.items()
                if not data["available"] or data["cosine"] is None
            ]
            if missing:
                print(f"  {model_name}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
