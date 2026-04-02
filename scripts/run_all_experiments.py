"""
Master experiment orchestrator for PersonaLens.
Runs all missing experiments with proper GPU allocation.

Usage:
    python scripts/run_all_experiments.py                    # Run everything
    python scripts/run_all_experiments.py --phase pipeline    # Only new model pipelines
    python scripts/run_all_experiments.py --phase ablations   # Only ablation experiments
    python scripts/run_all_experiments.py --phase localization # Only localization
    python scripts/run_all_experiments.py --phase bfi         # Only BFI V2
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONPATH"] = str(REPO_ROOT)

CONDA_ENV = "personaforge_env"
PYTHON = f"conda run --no-banner -n {CONDA_ENV} python -u"


def log(msg):
    print(f"\033[0;34m[{time.strftime('%H:%M:%S')}]\033[0m {msg}", flush=True)


def ok(msg):
    print(f"\033[0;32m[{time.strftime('%H:%M:%S')}] ✓\033[0m {msg}", flush=True)


def warn(msg):
    print(f"\033[1;33m[{time.strftime('%H:%M:%S')}] ⚠\033[0m {msg}", flush=True)


def err(msg):
    print(f"\033[0;31m[{time.strftime('%H:%M:%S')}] ✗\033[0m {msg}", flush=True)


def model_short(model):
    return model.replace("/", "_")


def result_exists(path_pattern):
    return os.path.exists(path_pattern)


def run(cmd, gpu=None, log_file=None):
    """Run command with optional GPU restriction. Returns success bool."""
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if gpu is not None:
        log(f"  GPU: {gpu}")

    full_cmd = f"conda run --no-banner -n {CONDA_ENV} python -u {' '.join(cmd)}"
    if log_file:
        full_cmd += f" 2>&1 | tee {log_file}"

    try:
        result = subprocess.run(
            full_cmd, shell=True, cwd=str(REPO_ROOT), env=env, check=False
        )
        if result.returncode != 0:
            err(f"Command failed with code {result.returncode}")
            return False
        return True
    except Exception as e:
        err(f"Exception: {e}")
        return False


BIG5 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

EXISTING_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/gemma-2-2b-it",
    "unsloth/Llama-3.2-1B-Instruct",
]

NEW_MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", 0),        # 24GB free on GPU 0
    ("mistralai/Mistral-7B-Instruct-v0.1", 3), # 19GB free on GPU 3
    ("Qwen/Qwen2.5-1.5B-Instruct", 5),        # 10GB free on GPU 5
]


def check_missing_pipeline(model):
    ms = model_short(model)
    missing = []
    for trait in BIG5:
        act_dir = REPO_ROOT / "results" / "activations" / ms / trait
        if not act_dir.exists() or not list(act_dir.glob("*.npy")):
            missing.append(f"activations/{trait}")
    vec_dir = REPO_ROOT / "results" / "persona_vectors" / ms
    if not vec_dir.exists():
        missing.append("persona_vectors")
    return missing


def check_missing_localization(model, traits=None):
    ms = model_short(model)
    if traits is None:
        traits = BIG5
    missing = []
    for trait in traits:
        result_file = REPO_ROOT / "results" / "localization" / ms / f"refined_{trait}.json"
        if not result_file.exists():
            missing.append(trait)
    return missing


def check_missing_bfi_v2(model):
    ms = model_short(model)
    missing = []
    for trait in BIG5:
        result_file = REPO_ROOT / "results" / "bfi_behavioral_v2" / ms / f"responses_{trait}.json"
        if not result_file.exists():
            missing.append(trait)
    return missing


def run_pipeline_phase():
    """Phase 1: Run full pipeline (activations + vectors + steering) for new models."""
    log("=" * 60)
    log("PHASE 1: New Model Pipelines")
    log("=" * 60)

    procs = {}
    for model, gpu in NEW_MODELS:
        missing = check_missing_pipeline(model)
        if not missing:
            warn(f"{model}: pipeline already complete, skipping")
            continue

        log(f"Starting pipeline for {model} on GPU {gpu}")
        ms = model_short(model)
        log_file = str(REPO_ROOT / "results" / "logs" / f"{ms}_pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = (
            f"conda run --no-banner -n {CONDA_ENV} python -u "
            f"scripts/run_pipeline.py --model {model} --trait big5 "
            f"--device cuda --skip_localize "
            f"2>&1 | tee {log_file}"
        )

        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(REPO_ROOT), env=env
        )
        procs[model] = (proc, gpu)

    if not procs:
        ok("No new model pipelines needed")
        return True

    log(f"Waiting for {len(procs)} pipeline(s)...")
    all_ok = True
    for model, (proc, gpu) in procs.items():
        ret = proc.wait()
        if ret == 0:
            ok(f"Pipeline complete: {model}")
        else:
            err(f"Pipeline FAILED: {model} (code {ret})")
            all_ok = False

    return all_ok


def run_gemma_localization():
    """Fix missing Gemma-2 localization (only openness exists)."""
    log("Checking Gemma-2 localization...")
    missing = check_missing_localization("unsloth/gemma-2-2b-it")
    if not missing:
        ok("Gemma-2 localization complete, skipping")
        return True

    log(f"Gemma-2 missing localization for: {missing}")
    for trait in missing:
        cmd = [
            "src/localization/localize_circuits_v2.py",
            "--model", "unsloth/gemma-2-2b-it",
            "--trait", trait,
            "--device", "cuda",
            "--n_samples", "10",
        ]
        ok_run = run(cmd, gpu=5)
        if not ok_run:
            err(f"Gemma-2 localization failed for {trait}")
            return False

    ok("Gemma-2 localization complete")
    return True


def run_localization_phase():
    """Phase 3: Localization for new models."""
    log("=" * 60)
    log("PHASE 3: Localization for new models")
    log("=" * 60)

    for model, gpu in NEW_MODELS:
        ms = model_short(model)
        missing = check_missing_localization(model)
        if not missing:
            warn(f"{model}: localization already complete, skipping")
            continue

        # Check that activations exist first
        act_dir = REPO_ROOT / "results" / "activations" / ms
        if not act_dir.exists():
            err(f"{model}: no activations found, run pipeline first")
            continue

        log(f"Running localization for {model} on GPU {gpu}")
        for trait in missing:
            cmd = [
                "src/localization/localize_circuits_v2.py",
                "--model", model,
                "--trait", trait,
                "--device", "cuda",
                "--n_samples", "10",
            ]
            ok_run = run(cmd, gpu=gpu)
            if not ok_run:
                err(f"Localization failed for {model}/{trait}")

    ok("Localization phase complete")
    return True


def run_ablation_phase(models=None):
    """Phase 2: Ablation experiments for specified models."""
    log("=" * 60)
    log("PHASE 2: Ablation Experiments")
    log("=" * 60)

    if models is None:
        models = EXISTING_MODELS[1:]  # Skip Qwen2.5-0.5B (already done)

    for model in models:
        ms = model_short(model)
        act_dir = REPO_ROOT / "results" / "activations" / ms

        if not act_dir.exists():
            warn(f"{model}: no activations, skipping ablations")
            continue

        # OOD Generalization (no model loading needed)
        ood_dir = REPO_ROOT / "results" / "ood_results" / ms
        if not ood_dir.exists():
            log(f"Running OOD for {model}...")
            cmd = [
                "src/evaluation/eval_ood_generalization.py",
                "--activations_dir", str(act_dir),
                "--trait", "all",
                "--output_dir", "results/ood_results",
            ]
            run(cmd, gpu=None)  # CPU-only
        else:
            warn(f"{model}: OOD already exists, skipping")

    # Shuffle + Null need model loading — run sequentially
    for model in models:
        ms = model_short(model)

        # Shuffle label baseline
        shuffle_dir = REPO_ROOT / "results" / "shuffle_label_baseline_results" / ms
        if not shuffle_dir.exists():
            log(f"Running shuffle baseline for {model}...")
            cmd = [
                "src/evaluation/eval_shuffle_label_baseline.py",
                "--model", model,
                "--n_permutations", "100",
            ]
            run(cmd, gpu=0)  # Use GPU 0 for sequential model loading
        else:
            warn(f"{model}: shuffle baseline already exists, skipping")

        # Null orthogonality
        null_dir = REPO_ROOT / "results" / "null_orthogonality_results" / ms
        if not null_dir.exists():
            log(f"Running null orthogonality for {model}...")
            cmd = [
                "src/evaluation/eval_null_orthogonality.py",
                "--model", model,
            ]
            run(cmd, gpu=0)
        else:
            warn(f"{model}: null orthogonality already exists, skipping")

    ok("Ablation phase complete")
    return True


def run_bfi_v2_phase():
    """Phase 4: BFI V2 behavioral evaluation for new models."""
    log("=" * 60)
    log("PHASE 4: BFI V2 Behavioral Evaluation")
    log("=" * 60)

    for model, gpu in NEW_MODELS:
        ms = model_short(model)
        missing = check_missing_bfi_v2(model)

        if not missing:
            warn(f"{model}: BFI V2 already complete, skipping")
            continue

        # Need persona vectors to exist
        vec_dir = REPO_ROOT / "results" / "persona_vectors" / ms
        if not vec_dir.exists():
            err(f"{model}: no persona vectors, run pipeline first")
            continue

        log(f"Running BFI V2 for {model} on GPU {gpu}")
        for trait in missing:
            cmd = [
                "src/evaluation/eval_bfi_behavioral_v2.py",
                "--model", model,
                "--trait", trait,
                "--device", "cuda",
            ]
            ok_run = run(cmd, gpu=gpu)
            if not ok_run:
                err(f"BFI V2 failed for {model}/{trait}")

    ok("BFI V2 phase complete")
    return True


def run_new_ablations():
    """Phase 5: Ablations for new models."""
    log("=" * 60)
    log("PHASE 5: Ablations for new models")
    log("=" * 60)

    new_model_names = [m for m, _ in NEW_MODELS]
    new_model_gpus = {m: g for m, g in NEW_MODELS}

    # OOD (CPU)
    for model in new_model_names:
        ms = model_short(model)
        act_dir = REPO_ROOT / "results" / "activations" / ms
        ood_dir = REPO_ROOT / "results" / "ood_results" / ms

        if not act_dir.exists():
            warn(f"{model}: no activations, skipping OOD")
            continue
        if ood_dir.exists():
            warn(f"{model}: OOD already exists, skipping")
            continue

        log(f"Running OOD for {model}...")
        cmd = [
            "src/evaluation/eval_ood_generalization.py",
            "--activations_dir", str(act_dir),
            "--trait", "all",
            "--output_dir", "results/ood_results",
        ]
        run(cmd)

    # Shuffle + Null (sequential, GPU 0)
    for model in new_model_names:
        ms = model_short(model)

        shuffle_dir = REPO_ROOT / "results" / "shuffle_label_baseline_results" / ms
        if not shuffle_dir.exists():
            log(f"Running shuffle baseline for {model}...")
            run([
                "src/evaluation/eval_shuffle_label_baseline.py",
                "--model", model,
                "--n_permutations", "100",
            ], gpu=0)

        null_dir = REPO_ROOT / "results" / "null_orthogonality_results" / ms
        if not null_dir.exists():
            log(f"Running null orthogonality for {model}...")
            run([
                "src/evaluation/eval_null_orthogonality.py",
                "--model", model,
            ], gpu=0)

    ok("New model ablations complete")
    return True


def print_summary():
    log("=" * 60)
    log("EXPERIMENT SUMMARY")
    log("=" * 60)

    all_models = EXISTING_MODELS + [m for m, _ in NEW_MODELS]

    for model in all_models:
        ms = model_short(model)
        log(f"\n  {model}:")

        for dirname in ["activations", "persona_vectors", "localization",
                         "steering_results", "bfi_behavioral_v2",
                         "ood_results", "shuffle_label_baseline_results",
                         "null_orthogonality_results"]:
            d = REPO_ROOT / "results" / dirname / ms
            if d.exists():
                n = len(list(d.iterdir()))
                log(f"    {dirname}: {n} items ✓")
            else:
                log(f"    {dirname}: MISSING ✗")


def main():
    parser = argparse.ArgumentParser(description="PersonaLens experiment orchestrator")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "pipeline", "ablations", "localization",
                                 "bfi", "new-ablations", "gemma-loc", "summary"])
    args = parser.parse_args()

    os.makedirs(REPO_ROOT / "results" / "logs", exist_ok=True)

    if args.phase == "summary":
        print_summary()
        return

    log("PersonaLens Experiment Orchestrator")
    log(f"Phase: {args.phase}")
    log(f"Repo: {REPO_ROOT}")

    success = True

    if args.phase in ("all", "pipeline"):
        if not run_pipeline_phase():
            success = False

    if args.phase in ("all", "gemma-loc"):
        if not run_gemma_localization():
            success = False

    if args.phase in ("all", "ablations"):
        if not run_ablation_phase():
            success = False

    if args.phase in ("all", "localization"):
        if not run_localization_phase():
            success = False

    if args.phase in ("all", "bfi"):
        if not run_bfi_v2_phase():
            success = False

    if args.phase in ("all", "new-ablations"):
        if not run_new_ablations():
            success = False

    print_summary()

    if success:
        ok("ALL EXPERIMENTS COMPLETE")
    else:
        err("SOME EXPERIMENTS FAILED - check logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
