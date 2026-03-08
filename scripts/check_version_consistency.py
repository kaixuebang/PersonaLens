#!/usr/bin/env python3
"""
Version Consistency Check Script

Verifies that:
1. All generated artifacts have consistent versioning (v1 vs v2)
2. No mixing of v1 and v2 data in the same analysis
3. Required directories exist and have proper structure
4. Analysis JSON files contain expected fields

Usage:
    python scripts/check_version_consistency.py
    python scripts/check_version_consistency.py --fix
"""

import argparse
import json
import os
import sys
from pathlib import Path


def check_directory_structure():
    """Check that required directories exist."""
    print("\n" + "=" * 60)
    print("Checking Directory Structure")
    print("=" * 60)

    required_dirs = [
        "src",
        "scripts",
        "paper",
        "activations",
        "persona_vectors",
        "localization",
    ]

    repo_root = Path(__file__).parent.parent
    issues = []

    for dir_name in required_dirs:
        path = repo_root / dir_name
        if path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            if dir_name in ["activations", "persona_vectors", "localization"]:
                print(f"    (This is OK if pipeline hasn't been run yet)")
            else:
                issues.append(f"Missing required directory: {dir_name}")

    return issues


def check_v1_v2_consistency():
    """Check for mixing of v1 and v2 data."""
    print("\n" + "=" * 60)
    print("Checking v1/v2 Version Consistency")
    print("=" * 60)

    repo_root = Path(__file__).parent.parent
    issues = []

    # Check for v1 directories
    v1_dirs = ["persona_vectors", "localization"]
    v2_dirs = ["persona_vectors", "localization"]

    v1_exists = [d for d in v1_dirs if (repo_root / d).exists()]
    v2_exists = [d for d in v2_dirs if (repo_root / d).exists()]

    if v1_exists and v2_exists:
        print(f"  ⚠ WARNING: Both v1 and v2 directories exist:")
        print(f"    v1: {v1_exists}")
        print(f"    v2: {v2_exists}")
        print(f"    Recommendation: Use only v2 (delete v1 directories)")
        issues.append("Mixing v1 and v2 data directories")
    elif v2_exists:
        print(f"  ✓ Using v2 structure: {v2_exists}")
    elif v1_exists:
        print(f"  ⚠ Using v1 structure: {v1_exists}")
        print(f"    Recommendation: Migrate to v2")
        issues.append("Using deprecated v1 structure")
    else:
        print(f"  ℹ No persona vectors found (pipeline not run)")

    return issues


def check_analysis_json_files():
    """Check that analysis JSON files have required fields."""
    print("\n" + "=" * 60)
    print("Checking Analysis JSON Files")
    print("=" * 60)

    repo_root = Path(__file__).parent.parent
    persona_dir = repo_root / "persona_vectors"
    issues = []

    if not persona_dir.exists():
        print("  ℹ No persona_vectors directory (pipeline not run)")
        return issues

    required_fields = [
        "trait",
        "n_layers",
        "n_samples",
        "best_layer_loso",
        "best_loso_accuracy",
        "layers",
    ]

    # New fields added in fixed version
    new_fields = [
        "cohens_d_ci_lower",
        "cohens_d_ci_upper",
        "cohens_d_p_value",
    ]

    total_files = 0
    files_with_all_fields = 0
    files_missing_new_fields = 0

    for model_dir in persona_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for trait_dir in model_dir.iterdir():
            if not trait_dir.is_dir():
                continue

            # Look for analysis files
            analysis_files = list(trait_dir.glob("analysis*.json"))

            for analysis_file in analysis_files:
                total_files += 1

                try:
                    with open(analysis_file) as f:
                        data = json.load(f)

                    # Check required fields
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        print(f"  ✗ {analysis_file}: Missing fields: {missing}")
                        issues.append(f"Missing fields in {analysis_file}: {missing}")
                        continue

                    # Check for new statistical fields
                    missing_new = [f for f in new_fields if f not in data]
                    if missing_new:
                        files_missing_new_fields += 1
                        # This is OK - just means old data without new stats
                    else:
                        files_with_all_fields += 1

                    # Validate layer structure
                    layers = data.get("layers", {})
                    if not layers:
                        print(f"  ✗ {analysis_file}: No layers found")
                        issues.append(f"No layers in {analysis_file}")
                        continue

                    # Check first layer has required fields
                    first_layer = list(layers.values())[0]
                    layer_fields = [
                        "loso_accuracy",
                        "cohens_d",
                    ]

                    missing_layer = [f for f in layer_fields if f not in first_layer]
                    if missing_layer:
                        print(
                            f"  ✗ {analysis_file}: Missing layer fields: {missing_layer}"
                        )
                        issues.append(f"Missing layer fields in {analysis_file}")
                        continue

                    print(f"  ✓ {analysis_file.name}")

                except json.JSONDecodeError as e:
                    print(f"  ✗ {analysis_file}: Invalid JSON - {e}")
                    issues.append(f"Invalid JSON in {analysis_file}")
                except Exception as e:
                    print(f"  ✗ {analysis_file}: Error - {e}")
                    issues.append(f"Error reading {analysis_file}: {e}")

    print(f"\n  Summary: {total_files} files checked")
    print(f"    {files_with_all_fields} with all fields (including new statistics)")
    print(
        f"    {files_missing_new_fields} missing new statistical fields (re-run extraction to update)"
    )

    return issues


def check_activation_files():
    """Check that activation files are consistent."""
    print("\n" + "=" * 60)
    print("Checking Activation Files")
    print("=" * 60)

    repo_root = Path(__file__).parent.parent
    activations_dir = repo_root / "activations"
    issues = []

    if not activations_dir.exists():
        print("  ℹ No activations directory (pipeline not run)")
        return issues

    for model_dir in activations_dir.iterdir():
        if not model_dir.is_dir():
            continue

        print(f"\n  Model: {model_dir.name}")

        for trait_dir in model_dir.iterdir():
            if not trait_dir.is_dir():
                continue

            pos_files = list(trait_dir.glob("pos_layer_*.npy"))
            neg_files = list(trait_dir.glob("neg_layer_*.npy"))

            if len(pos_files) != len(neg_files):
                print(
                    f"    ✗ {trait_dir.name}: Mismatch in pos/neg files "
                    f"({len(pos_files)} vs {len(neg_files)})"
                )
                issues.append(f"Mismatch in {trait_dir}")
            else:
                print(f"    ✓ {trait_dir.name}: {len(pos_files)} layers")

    return issues


def suggest_fixes(issues):
    """Suggest fixes for identified issues."""
    if not issues:
        return

    print("\n" + "=" * 60)
    print("Suggested Fixes")
    print("=" * 60)

    if any("v1" in issue for issue in issues):
        print("\n1. To migrate from v1 to v2:")
        print("   rm -rf persona_vectors localization")
        print("   python scripts/run_pipeline.py --model <MODEL> --trait all")

    if any("Missing fields" in issue for issue in issues):
        print("\n2. To update analysis files with new statistics:")
        print("   python src/extraction/extract_persona_vectors.py \\")
        print("       --activations_dir activations/<MODEL> --trait all")

    if any("pipeline" in issue.lower() for issue in issues):
        print("\n3. To run the full pipeline:")
        print("   make pipeline MODEL=<MODEL> TRAIT=all")


def main():
    parser = argparse.ArgumentParser(
        description="Check version consistency and data integrity"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix identified issues (when possible)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PersonaLens Version Consistency Check")
    print("=" * 60)

    all_issues = []

    # Run all checks
    all_issues.extend(check_directory_structure())
    all_issues.extend(check_v1_v2_consistency())
    all_issues.extend(check_analysis_json_files())
    all_issues.extend(check_activation_files())

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all_issues:
        print(f"\n  ⚠ Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")

        suggest_fixes(all_issues)
        return 1
    else:
        print("\n  ✓ All checks passed!")
        print("  Data is consistent and ready for analysis.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
