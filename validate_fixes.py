#!/usr/bin/env python3
"""
Validation Experiment Runner
Runs a complete end-to-end experiment to prove the fixes work.
"""

import os
import sys
import json
import time
from pathlib import Path

# Test configuration
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIT = "openness"
DEVICE = "cuda"


def log(msg):
    print(f"\n[VALIDATION] {msg}")
    sys.stdout.flush()


def check_file(path, description):
    """Check if a file exists and report."""
    if Path(path).exists():
        size = Path(path).stat().st_size
        log(f"✓ {description}: {path} ({size} bytes)")
        return True
    else:
        log(f"✗ {description}: {path} NOT FOUND")
        return False


def main():
    log("=" * 60)
    log("STARTING VALIDATION EXPERIMENT")
    log("=" * 60)
    log(f"Model: {MODEL}")
    log(f"Trait: {TRAIT}")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    start_time = time.time()

    # Step 1: Environment check
    log("Step 1: Environment Verification")
    import torch
    import transformers

    log(f"  PyTorch: {torch.__version__}")
    log(f"  Transformers: {transformers.__version__}")
    log(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Step 2: Run pipeline
    log("Step 2: Running Full Pipeline")
    log("  This will take ~5-10 minutes on RTX 4060...")

    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--model",
        MODEL,
        "--trait",
        TRAIT,
        "--device",
        DEVICE,
        "--skip_localize",  # Skip slow patching for quick validation
    ]

    log(f"  Command: {' '.join(cmd)}")

    import subprocess

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        log("✗ PIPELINE FAILED")
        return 1

    log("✓ Pipeline completed successfully")

    # Step 3: Verify outputs
    log("Step 3: Verifying Output Files")

    model_short = MODEL.replace("/", "_")
    checks = []

    # Check activations
    checks.append(
        check_file(
            f"activations/{model_short}/{TRAIT}/pos_layer_0.npy",
            "Activation file (pos)",
        )
    )
    checks.append(
        check_file(
            f"activations/{model_short}/{TRAIT}/neg_layer_0.npy",
            "Activation file (neg)",
        )
    )

    # Check persona vectors
    checks.append(
        check_file(
            f"persona_vectors_v2/{model_short}/{TRAIT}/vectors/mean_diff_layer_0.npy",
            "Persona vector (mean_diff)",
        )
    )
    checks.append(
        check_file(
            f"persona_vectors_v2/{model_short}/{TRAIT}/analysis_v2_{TRAIT}.json",
            "Analysis JSON",
        )
    )

    # Check steering results
    checks.append(
        check_file(
            f"steering_results/{model_short}/{TRAIT}/comparison_{TRAIT}_alpha3.0.json",
            "Steering results",
        )
    )

    if not all(checks):
        log("✗ Some output files are missing!")
        return 1

    # Step 4: Verify new statistical fields
    log("Step 4: Checking Statistical Rigor (New Fields)")

    analysis_file = f"persona_vectors_v2/{model_short}/{TRAIT}/analysis_v2_{TRAIT}.json"
    with open(analysis_file) as f:
        data = json.load(f)

    # Check top-level fields
    required_fields = [
        "trait",
        "n_layers",
        "n_samples",
        "best_layer_loso",
        "best_loso_accuracy",
        "best_layer_snr",
    ]

    new_stat_fields = ["cohens_d_ci_lower", "cohens_d_ci_upper", "cohens_d_p_value"]

    log("  Checking required fields:")
    for field in required_fields:
        if field in data:
            log(f"    ✓ {field}: {data[field]}")
        else:
            log(f"    ✗ {field}: MISSING")
            return 1

    log("  Checking NEW statistical fields (post-audit fix):")
    layers_with_new_stats = 0
    for layer_idx, layer_data in data.get("layers", {}).items():
        has_all = all(f in layer_data for f in new_stat_fields)
        if has_all:
            layers_with_new_stats += 1
            if layer_idx == "0" or layer_idx == 0:
                log(f"    ✓ Layer {layer_idx} has bootstrap CI and p-values:")
                log(f"      Cohen's d: {layer_data['cohens_d']:.2f}")
                log(
                    f"      95% CI: [{layer_data['cohens_d_ci_lower']:.2f}, {layer_data['cohens_d_ci_upper']:.2f}]"
                )
                log(f"      p-value: {layer_data['cohens_d_p_value']:.4f}")

    if layers_with_new_stats == 0:
        log("  ✗ No layers found with new statistical fields!")
        return 1

    log(f"  ✓ {layers_with_new_stats} layers have bootstrap confidence intervals")

    # Step 5: Generate tables
    log("Step 5: Generating LaTeX Tables")

    cmd = [
        sys.executable,
        "scripts/generate_latex_tables.py",
        "--persona_vectors_dir",
        "persona_vectors_v2",
        "--output_dir",
        "paper/tables",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log("✓ Tables generated successfully")
        # Show what was generated
        if Path("paper/tables").exists():
            files = list(Path("paper/tables").glob("*.tex"))
            log(f"  Generated {len(files)} table files:")
            for f in files:
                log(f"    - {f.name}")
    else:
        log("✗ Table generation failed")
        log(result.stderr)
        return 1

    # Final summary
    elapsed = time.time() - start_time
    log("=" * 60)
    log("VALIDATION COMPLETE - ALL CHECKS PASSED")
    log("=" * 60)
    log(f"Elapsed time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    log("")
    log("Key Results:")
    log(f"  - Model: {MODEL}")
    log(f"  - Trait: {TRAIT}")
    log(f"  - Best layer: {data['best_layer_loso']}")
    log(f"  - Best LOSO accuracy: {data['best_loso_accuracy']:.3f}")
    log(f"  - Output files: {sum(checks)}/{len(checks)} verified")
    log(f"  - Statistical rigor: Bootstrap CI and p-values present")
    log("")
    log("✓ Fixes validated - the pipeline now:")
    log("  1. Runs without 'System role' errors")
    log("  2. Generates all required output files")
    log("  3. Includes 95% confidence intervals for Cohen's d")
    log("  4. Includes permutation test p-values")
    log("  5. Auto-generates LaTeX tables from results")
    log("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
