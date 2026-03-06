"""
run_pipeline.py - One-command full pipeline runner

Runs the complete personality neuron extraction and steering pipeline:
1. Collect activations from contrastive prompt pairs
2. Extract persona vectors (mean diff, PCA, linear probe)
3. Causal localization via activation patching
4. Personality steering demonstration

Usage:
    # Full pipeline for one trait
    python run_pipeline.py --model Qwen/Qwen3-0.6B --trait openness

    # Full pipeline for all Big Five traits
    python run_pipeline.py --model Qwen/Qwen3-0.6B --trait big5

    # Full pipeline for all traits
    python run_pipeline.py --model Qwen/Qwen3-0.6B --trait all

    # Skip collection (if already done), only analyze + steer
    python run_pipeline.py --model Qwen/Qwen3-0.6B --trait openness --skip_collect
"""

import argparse
import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.prompts.contrastive_prompts import get_all_trait_names, BIG_FIVE_PROMPTS, DEFENSE_MECHANISM_PROMPTS


def run_step(cmd, step_name):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*70}")
    print(f"  STEP: {step_name}")
    print(f"{'='*70}")
    print(f"  CMD: {' '.join(cmd)}\n")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Add repo_root to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {step_name}")
        return False
    print(f"\n  ✓ COMPLETED: {step_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete personality analysis pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trait", type=str, default="openness",
                        help="Trait name, 'big5', 'defense', or 'all'")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="Steering strength for demo")
    parser.add_argument("--skip_collect", action="store_true",
                        help="Skip activation collection (use existing)")
    parser.add_argument("--skip_localize", action="store_true",
                        help="Skip causal localization (slow)")
    parser.add_argument("--n_patching_samples", type=int, default=5,
                        help="Number of samples for activation patching")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Resolve trait list
    if args.trait == "big5":
        traits = list(BIG_FIVE_PROMPTS.keys())
    elif args.trait == "defense":
        traits = list(DEFENSE_MECHANISM_PROMPTS.keys())
    elif args.trait == "all":
        traits = get_all_trait_names()
    else:
        traits = [args.trait]

    model_short = args.model.replace("/", "_")
    python = sys.executable

    print(f"\n{'#'*70}")
    print(f"  PersonaForge 2.0 - Personality Neuron Analysis Pipeline")
    print(f"  Model: {args.model}")
    print(f"  Traits: {traits}")
    print(f"{'#'*70}")

    device_args = ["--device", args.device] if args.device else []

    # Step 1: Collect Activations
    if not args.skip_collect:
        for trait in traits:
            success = run_step(
                [python, "src/localization/collect_activations.py",
                 "--model", args.model,
                 "--trait", trait,
                 "--output_dir", "activations"] + device_args,
                f"Collect Activations - {trait}"
            )
            if not success:
                print(f"Stopping pipeline due to failure in collection for {trait}")
                return

    # Step 2: Extract Persona Vectors
    success = run_step(
        [python, "src/extraction/extract_persona_vectors_v2.py",
         "--activations_dir", f"activations/{model_short}",
         "--trait", "all"],
        "Extract Persona Vectors"
    )
    if not success:
        print("Stopping pipeline due to failure in extraction")
        return

    # Step 3: Causal Localization (optional, slow)
    if not args.skip_localize:
        for trait in traits:
            run_step(
                [python, "src/localization/localize_circuits_v2.py",
                 "--model", args.model,
                 "--trait", trait,
                 "--n_samples", str(args.n_patching_samples)] + device_args,
                f"Causal Localization - {trait}"
            )

    # Step 4: Personality Steering Demo
    for trait in traits:
        run_step(
            [python, "src/steering/steer_personality.py",
             "--model", args.model,
             "--trait", trait,
             "--alpha", str(args.alpha),
             "--vectors_dir", f"persona_vectors/{model_short}/{trait}/vectors",
             "--vector_type", "mean_diff"] + device_args,
            f"Steering Demo - {trait}"
        )

    # Step 5: Cross-Model Validation (if applicable)
    run_step(
        [python, "src/evaluation/cross_model_validation.py",
         "--persona_vectors_dir", "persona_vectors_v2",
         "--trait", "all"],
        "Cross-Model Validation"
    )

    print(f"\n{'#'*70}")
    print(f"  ✓ PIPELINE COMPLETE")
    print(f"  Activations:     activations/{model_short}/")
    print(f"  Persona Vectors: persona_vectors/{model_short}/")
    print(f"  Localization:    localization/{model_short}/")
    print(f"  Steering:        steering_results/{model_short}/")
    print(f"  Cross-Model:     cross_model_results/")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
