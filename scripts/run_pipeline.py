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
import atexit
import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.prompts.contrastive_prompts import (
    get_all_trait_names,
    BIG_FIVE_PROMPTS,
    DEFENSE_MECHANISM_PROMPTS,
)


def check_dependencies():
    """Pre-flight check: Verify all required dependencies are installed."""
    print("\n" + "=" * 50)
    print("Pre-flight Checks")
    print("=" * 50)

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
    ]

    missing = []
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    if missing:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing)}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

    # Check CUDA availability
    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ CUDA not available - will use CPU (slower)")

    # Check directory structure
    required_dirs = ["src", "scripts", "paper"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in required_dirs:
        path = os.path.join(repo_root, d)
        if os.path.isdir(path):
            print(f"  ✓ Directory: {d}/")
        else:
            print(f"  ✗ Directory: {d}/ - MISSING")
            return False

    print("  ✓ All pre-flight checks passed")
    print("=" * 50 + "\n")
    return True


def verify_outputs(model_short):
    """Post-flight check: Verify all expected outputs were generated."""
    print("\n" + "=" * 50)
    print("Verifying Pipeline Outputs")
    print("=" * 50)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))

    expected_paths = [
        f"results/activations/{model_short}",
        f"results/persona_vectors/{model_short}",
        f"results/localization/{model_short}",
        f"results/steering_results/{model_short}",
        "results/cross_model_results",
    ]

    all_exist = True
    for path in expected_paths:
        full_path = os.path.join(repo_root, path)
        if os.path.isdir(full_path):
            # Count files in directory
            n_files = sum(1 for _ in os.walk(full_path) for _ in _)
            print(f"  ✓ {path}/ ({n_files} items)")
        else:
            print(f"  ✗ {path}/ - MISSING")
            all_exist = False

    print("=" * 50 + "\n")
    return all_exist


def run_step(name, cmd, cwd=None, env=None):
    """Run a pipeline step with error handling."""
    print(f"\n{'=' * 50}")
    print(f"Running step: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 50}\n")
    try:
        result = subprocess.run(cmd, check=True, cwd=cwd, env=env, capture_output=False)
        print(f"\n  ✓ Step '{name}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step '{name}' failed with return code {e.returncode}.")
        print("Pipeline aborted to prevent false completion states.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete personality analysis pipeline"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--trait",
        type=str,
        default="openness",
        help="Trait name, 'big5', 'defense', or 'all'",
    )
    parser.add_argument(
        "--alpha", type=float, default=3.0, help="Steering strength for demo"
    )
    parser.add_argument(
        "--skip_collect",
        action="store_true",
        help="Skip activation collection (use existing)",
    )
    parser.add_argument(
        "--skip_localize", action="store_true", help="Skip causal localization (slow)"
    )
    parser.add_argument(
        "--n_patching_samples",
        type=int,
        default=5,
        help="Number of samples for activation patching",
    )
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

    print(f"\n{'#' * 70}")
    print(f"  PersonaForge 2.0 - Personality Neuron Analysis Pipeline")
    print(f"  Model: {args.model}")
    print(f"  Traits: {traits}")
    print(f"{'#' * 70}")

    # Pre-flight checks
    if not check_dependencies():
        sys.exit(1)

    # Verify outputs on completion
    atexit.register(lambda: verify_outputs(model_short))

    device_args = ["--device", args.device] if args.device else []

    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    # Step 1: Collect Activations
    if not args.skip_collect:
        for trait in traits:
            run_step(
                f"Collect Activations - {trait}",
                [
                    python,
                    "src/localization/collect_activations.py",
                    "--model",
                    args.model,
                    "--trait",
                    trait,
                    "--output_dir",
                    "results/activations",
                ]
                + device_args,
                cwd=repo_root,
                env=env,
            )

    # Step 2: Extract Persona Vectors
    run_step(
        "Extract Persona Vectors",
        [
            python,
            "src/extraction/extract_persona_vectors_v2.py",
            "--activations_dir",
            f"results/activations/{model_short}",
            "--trait",
            "all",
        ],
        cwd=repo_root,
        env=env,
    )

    # Step 3: Causal Localization (optional, slow)
    if not args.skip_localize:
        for trait in traits:
            run_step(
                f"Causal Localization - {trait}",
                [
                    python,
                    "src/localization/localize_circuits_v2.py",
                    "--model",
                    args.model,
                    "--trait",
                    trait,
                    "--n_samples",
                    str(args.n_patching_samples),
                ]
                + device_args,
                cwd=repo_root,
                env=env,
            )

    # Step 4: Personality Steering Demo
    for trait in traits:
        run_step(
            f"Steering Demo - {trait}",
            [
                python,
                "src/steering/steer_personality.py",
                "--model",
                args.model,
                "--trait",
                trait,
                "--alpha",
                str(args.alpha),
                "--vectors_dir",
                f"results/persona_vectors/{model_short}/{trait}/vectors",
                "--vector_type",
                "mean_diff",
            ]
            + device_args,
            cwd=repo_root,
            env=env,
        )

    # Step 5: Cross-Model Validation (if applicable)
    run_step(
        "Cross-Model Validation",
        [
            python,
            "src/evaluation/cross_model_validation.py",
            "--persona_vectors_dir",
            "results/persona_vectors",
            "--trait",
            "all",
        ],
        cwd=repo_root,
        env=env,
    )

    print(f"\n{'#' * 70}")
    print(f"  ✓ PIPELINE COMPLETE")
    print(f"  Activations:     results/activations/{model_short}/")
    print(f"  Persona Vectors: results/persona_vectors/{model_short}/")
    print(f"  Localization:    results/localization/{model_short}/")
    print(f"  Steering:        results/steering_results/{model_short}/")
    print(f"  Cross-Model:     results/cross_model_results/")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
