import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict


class DeepPatternAnalyzer:
    """
    Deep analysis of personality representation patterns across models and layers.
    Extracts insights beyond basic metrics.
    """

    def __init__(self, results_dir: str = "results/persona_vectors"):
        self.results_dir = Path(results_dir)
        self.models = self._discover_models()

    def _discover_models(self) -> List[str]:
        """Auto-discover all models with results."""
        if not self.results_dir.exists():
            return []
        return [d.name for d in self.results_dir.iterdir() if d.is_dir()]

    def analyze_orthogonality_evolution(self) -> Dict:
        """
        Analyze how trait orthogonality evolves across layers.

        Returns:
            Dict with per-layer orthogonality matrices and trends
        """
        results = {}

        for model in self.models:
            model_results = {"layers": {}, "trend": []}

            # Check which layers have data for all Big Five traits
            traits = [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]

            # Load analysis to find available layers
            try:
                with open(
                    self.results_dir / model / "openness" / "analysis_v2_openness.json"
                ) as f:
                    analysis = json.load(f)
                    available_layers = [int(k) for k in analysis["layers"].keys()]
            except:
                continue

            for layer in available_layers:
                # Load vectors for all traits at this layer
                vectors = {}
                for trait in traits:
                    vec_path = (
                        self.results_dir
                        / model
                        / trait
                        / "vectors"
                        / f"mean_diff_layer_{layer}.npy"
                    )
                    if vec_path.exists():
                        v = np.load(vec_path)
                        vectors[trait] = v / np.linalg.norm(v)

                if len(vectors) == 5:
                    # Compute orthogonality matrix
                    trait_names = sorted(vectors.keys())
                    V = np.vstack([vectors[t] for t in trait_names])
                    sim_matrix = V @ V.T

                    # Extract off-diagonal |cos| values
                    off_diag = []
                    for i in range(5):
                        for j in range(i + 1, 5):
                            off_diag.append(abs(sim_matrix[i, j]))

                    model_results["layers"][layer] = {
                        "mean_cos": np.mean(off_diag),
                        "max_cos": np.max(off_diag),
                        "min_cos": np.min(off_diag),
                        "std_cos": np.std(off_diag),
                        "matrix": sim_matrix.tolist(),
                    }
                    model_results["trend"].append((layer, np.mean(off_diag)))

            if model_results["layers"]:
                results[model] = model_results

        return results

    def find_trait_interaction_patterns(self) -> Dict:
        """
        Identify patterns in how traits interact (correlation structure).

        Returns:
            Dict with trait correlation patterns across models
        """
        patterns = defaultdict(lambda: defaultdict(list))

        trait_pairs = [
            ("openness", "conscientiousness"),
            ("openness", "extraversion"),
            ("openness", "agreeableness"),
            ("openness", "neuroticism"),
            ("conscientiousness", "extraversion"),
            ("conscientiousness", "agreeableness"),
            ("conscientiousness", "neuroticism"),
            ("extraversion", "agreeableness"),
            ("extraversion", "neuroticism"),
            ("agreeableness", "neuroticism"),
        ]

        for model in self.models:
            # Find best layer for this model
            try:
                with open(
                    self.results_dir / model / "openness" / "analysis_v2_openness.json"
                ) as f:
                    analysis = json.load(f)
                    best_layer = analysis["best_layer_loso"]
            except:
                continue

            # Load vectors at best layer
            vectors = {}
            for trait in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]:
                vec_path = (
                    self.results_dir
                    / model
                    / trait
                    / "vectors"
                    / f"mean_diff_layer_{best_layer}.npy"
                )
                if vec_path.exists():
                    v = np.load(vec_path)
                    vectors[trait] = v / np.linalg.norm(v)

            # Compute pair similarities
            for t1, t2 in trait_pairs:
                if t1 in vectors and t2 in vectors:
                    cos_sim = abs(vectors[t1] @ vectors[t2])
                    patterns[f"{t1}-{t2}"][model] = cos_sim

        # Summarize patterns
        summary = {}
        for pair, model_values in patterns.items():
            if model_values:
                values = list(model_values.values())
                summary[pair] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "across_models": dict(model_values),
                }

        return summary

    def analyze_layer_progression_patterns(self) -> Dict:
        """
        Analyze how signal quality (LOSO, Cohen's d, SNR) progresses through layers.

        Returns:
            Dict with progression patterns per model
        """
        patterns = {}

        for model in self.models:
            model_patterns = {}

            for trait in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]:
                try:
                    analysis_path = (
                        self.results_dir / model / trait / f"analysis_v2_{trait}.json"
                    )
                    with open(analysis_path) as f:
                        analysis = json.load(f)

                    layers_data = []
                    for layer_str, metrics in analysis["layers"].items():
                        layer = int(layer_str)
                        layers_data.append(
                            {
                                "layer": layer,
                                "loso": metrics["loso_accuracy"],
                                "d": metrics["cohens_d"],
                                "snr": metrics["signal_to_noise_ratio"],
                                "diff_norm": metrics["diff_norm"],
                            }
                        )

                    # Sort by layer
                    layers_data.sort(key=lambda x: x["layer"])

                    # Detect patterns
                    loso_vals = [x["loso"] for x in layers_data]
                    d_vals = [x["d"] for x in layers_data]

                    model_patterns[trait] = {
                        "early_peak": self._detect_early_peak(loso_vals),
                        "monotonic_increase": self._detect_monotonic(d_vals),
                        "saturation_layer": self._find_saturation(loso_vals),
                        "layers_data": layers_data,
                    }

                except Exception as e:
                    print(f"Error analyzing {model}/{trait}: {e}")
                    continue

            if model_patterns:
                patterns[model] = model_patterns

        return patterns

    def _detect_early_peak(self, values: List[float]) -> bool:
        """Detect if values peak early then decline."""
        if len(values) < 3:
            return False
        peak_idx = np.argmax(values)
        return peak_idx < len(values) * 0.4  # Peak in first 40% of layers

    def _detect_monotonic(self, values: List[float]) -> bool:
        """Detect if values increase monotonically."""
        increases = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
        return increases / (len(values) - 1) > 0.7  # 70% increasing

    def _find_saturation(self, values: List[float]) -> int:
        """Find layer where values saturate (stop improving)."""
        if len(values) < 3:
            return 0
        threshold = 0.95 * max(values)
        for i, v in enumerate(values):
            if v >= threshold:
                return i
        return len(values) - 1

    def generate_insights_report(self) -> str:
        """Generate comprehensive insights report."""
        report = []
        report.append("=" * 70)
        report.append("DEEP PATTERN ANALYSIS REPORT")
        report.append("=" * 70)
        report.append()

        # Orthogonality evolution
        report.append("1. ORTHOGONALITY EVOLUTION ACROSS LAYERS")
        report.append("-" * 70)
        ortho_data = self.analyze_orthogonality_evolution()
        for model, data in ortho_data.items():
            if data["trend"]:
                layers, cos_vals = zip(*data["trend"])
                min_cos = min(cos_vals)
                max_cos = max(cos_vals)
                report.append(f"\n{model}:")
                report.append(f"  Range: {min_cos:.3f} - {max_cos:.3f}")
                report.append(
                    f"  Trend: {'Decreasing' if cos_vals[0] > cos_vals[-1] else 'Increasing'}"
                )

        report.append("\n\n2. TRAIT INTERACTION PATTERNS")
        report.append("-" * 70)
        interactions = self.find_trait_interaction_patterns()

        # Find most and least orthogonal pairs
        if interactions:
            sorted_pairs = sorted(interactions.items(), key=lambda x: x[1]["mean"])
            report.append("\nMost orthogonal (independent) trait pairs:")
            for pair, data in sorted_pairs[:3]:
                report.append(
                    f"  {pair}: |cos| = {data['mean']:.3f} ± {data['std']:.3f}"
                )

            report.append("\nLeast orthogonal (correlated) trait pairs:")
            for pair, data in sorted_pairs[-3:]:
                report.append(
                    f"  {pair}: |cos| = {data['mean']:.3f} ± {data['std']:.3f}"
                )

        report.append("\n\n3. LAYER PROGRESSION PATTERNS")
        report.append("-" * 70)
        progression = self.analyze_layer_progression_patterns()

        for model, traits in progression.items():
            report.append(f"\n{model}:")
            early_peak_traits = [t for t, d in traits.items() if d.get("early_peak")]
            if early_peak_traits:
                report.append(f"  Early peak traits: {', '.join(early_peak_traits)}")

            mono_traits = [t for t, d in traits.items() if d.get("monotonic_increase")]
            if mono_traits:
                report.append(f"  Monotonic increase traits: {', '.join(mono_traits)}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def main():
    """Run deep analysis on current results."""
    analyzer = DeepPatternAnalyzer()

    print(f"Discovered models: {analyzer.models}")
    print()

    report = analyzer.generate_insights_report()
    print(report)

    # Save report
    output_path = Path("results/analysis_insights.txt")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
