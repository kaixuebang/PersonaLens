#!/usr/bin/env python3
"""
Cross-Framework Orthogonality Analysis
Compare Big Five vs MBTI vs Jungian cognitive functions
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns


def load_vectors(model_dir, trait, layer):
    """Load normalized vector for a trait at specific layer."""
    vec_path = model_dir / trait / "vectors" / f"mean_diff_layer_{layer}.npy"
    if not vec_path.exists():
        return None
    v = np.load(vec_path)
    return v / np.linalg.norm(v)


def compute_orthogonality_matrix(vectors_dict):
    """Compute pairwise |cos| matrix."""
    traits = sorted(vectors_dict.keys())
    n = len(traits)
    matrix = np.zeros((n, n))

    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            if i == j:
                matrix[i, j] = 1.0
            else:
                cos_sim = np.dot(vectors_dict[t1], vectors_dict[t2])
                matrix[i, j] = abs(cos_sim)

    return matrix, traits


def analyze_framework(model_name, framework_traits, framework_name):
    """Analyze orthogonality for a specific framework."""
    model_dir = Path(f"results/persona_vectors/{model_name}")

    # Find common layer (use best layer from first trait as reference)
    first_trait = framework_traits[0]
    try:
        with open(model_dir / first_trait / f"analysis_v2_{first_trait}.json") as f:
            data = json.load(f)
            common_layer = data["best_layer_loso"]
    except:
        return None

    # Load all vectors
    vectors = {}
    for trait in framework_traits:
        v = load_vectors(model_dir, trait, common_layer)
        if v is not None:
            vectors[trait] = v

    if len(vectors) < 2:
        return None

    # Compute matrix
    matrix, trait_names = compute_orthogonality_matrix(vectors)

    # Extract off-diagonal values
    off_diag = []
    for i in range(len(trait_names)):
        for j in range(i + 1, len(trait_names)):
            off_diag.append(matrix[i, j])

    return {
        "framework": framework_name,
        "model": model_name,
        "layer": common_layer,
        "n_traits": len(trait_names),
        "matrix": matrix,
        "traits": trait_names,
        "mean_ortho": np.mean(off_diag),
        "std_ortho": np.std(off_diag),
        "min_ortho": np.min(off_diag),
        "max_ortho": np.max(off_diag),
        "off_diagonal": off_diag,
    }


def main():
    model = "Qwen_Qwen2.5-0.5B-Instruct"

    # Define framework traits
    frameworks = {
        "Big Five": [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ],
        "MBTI": ["extraversion_mbti", "sensing", "thinking", "judging"],
        "Jungian": ["ni", "ne", "si", "se", "ti", "te", "fi", "fe"],
    }

    print("=" * 80)
    print("CROSS-FRAMEWORK ORTHOGONALITY ANALYSIS")
    print(f"Model: {model}")
    print("=" * 80)
    print()

    results = {}
    for fw_name, traits in frameworks.items():
        result = analyze_framework(model, traits, fw_name)
        if result:
            results[fw_name] = result

            print(f"{fw_name} ({result['n_traits']} traits, Layer {result['layer']}):")
            print(
                f"  Mean |cos|: {result['mean_ortho']:.4f} ± {result['std_ortho']:.4f}"
            )
            print(f"  Range: [{result['min_ortho']:.4f}, {result['max_ortho']:.4f}]")
            print()

    # Cross-framework comparison
    if len(results) >= 2:
        print("=" * 80)
        print("CROSS-FRAMEWORK COMPARISON")
        print("=" * 80)
        print()

        # Sort by orthogonality (lower = more orthogonal)
        sorted_fw = sorted(results.items(), key=lambda x: x[1]["mean_ortho"])

        print("Ranking (most orthogonal = best separation):")
        for i, (fw_name, data) in enumerate(sorted_fw, 1):
            print(
                f"  {i}. {fw_name:12s}: {data['mean_ortho']:.4f} ± {data['std_ortho']:.4f}"
            )

        print()
        print("Key Insight:")
        best_fw = sorted_fw[0][0]
        print(f"  → {best_fw} shows the cleanest geometric structure")
        print(f"  → Lower |cos| = more independent dimensions")

    # Save matrices for visualization
    output_dir = Path("results/analysis")
    output_dir.mkdir(exist_ok=True)

    for fw_name, data in results.items():
        np.save(
            output_dir
            / f"ortho_matrix_{fw_name.lower().replace(' ', '_')}_{model}.npy",
            data["matrix"],
        )

    print()
    print(f"Matrices saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
