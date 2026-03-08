#!/usr/bin/env python3
"""
Cleanup script to remove old v1 directories and consolidate to clean structure.

This script will:
1. Remove old v1 directories (persona_vectors/, localization/, eval_results/)
2. Keep v2 directories as the canonical versions
3. Clean up any inconsistent data in persona_vectors/
4. Update .gitignore to prevent future confusion
"""

import os
import shutil
import sys
from pathlib import Path


def log(msg):
    print(f"[CLEANUP] {msg}")


def get_size(path):
    """Get total size of directory in MB."""
    total = 0
    if os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    log("=" * 60)
    log("PERSONALENS DIRECTORY CLEANUP")
    log("=" * 60)

    # Directories to remove (old v1 versions)
    dirs_to_remove = [
        "persona_vectors",  # Old v1 persona vectors
        "localization",  # Old v1 localization
        "eval_results",  # Old v1 eval results
        "activations",  # Confusing duplicate (activations/ is canonical)
    ]

    # Directories to keep (v2 is canonical)
    dirs_to_keep = [
        "activations",  # Canonical activation directory
        "persona_vectors",  # Canonical vector directory (code uses this name)
        "localization",  # Canonical localization directory
        "eval_results",  # Canonical eval directory
        "steering_results",  # Canonical steering directory
        "cross_model_results",  # Canonical cross-model directory
        "ood_results",  # Keep OOD results
    ]

    log("\nStep 1: Removing old v1 directories...")
    total_freed = 0
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            size = get_size(dir_name)
            log(f"  Removing: {dir_name}/ ({size:.1f} MB)")
            shutil.rmtree(dir_name)
            total_freed += size
        else:
            log(f"  Already removed: {dir_name}/")

    log(f"\n  Total space freed: {total_freed:.1f} MB")

    log("\nStep 2: Verifying v2 directories...")
    for dir_name in dirs_to_keep:
        if os.path.exists(dir_name):
            size = get_size(dir_name)
            count = sum(1 for _ in Path(dir_name).rglob("*") if _.is_file())
            log(f"  ✓ {dir_name}/ ({count} files, {size:.1f} MB)")
        else:
            log(f"  ℹ {dir_name}/ does not exist (will be created on run)")

    log("\nStep 3: Cleaning up inconsistent data...")

    # Check persona_vectors for incorrectly organized data
    pv2_dir = Path("persona_vectors")
    if pv2_dir.exists():
        # Should have model directories, not trait directories at top level
        trait_names = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "humor",
            "projection",
            "rationalization",
            "denial",
            "displacement",
            "intellectualization",
            "regression",
            "reaction_formation",
            "sublimation",
        ]

        removed_count = 0
        for item in pv2_dir.iterdir():
            if item.is_dir() and item.name in trait_names:
                log(f"  Removing misplaced trait dir: {item.name}/")
                shutil.rmtree(item)
                removed_count += 1

        if removed_count == 0:
            log("  No misplaced directories found")
        else:
            log(f"  Removed {removed_count} misplaced directories")

    log("\nStep 4: Creating .cleanup_done marker...")
    with open(".cleanup_done", "w") as f:
        f.write("Directory cleanup completed. v2 directories are canonical.\n")
        f.write(f"Freed space: {total_freed:.1f} MB\n")

    log("\nStep 5: Updating .gitignore...")
    gitignore_additions = """
# Generated outputs (v2 canonical)
activations/*
persona_vectors/*
localization/*
eval_results/*
!activations/.gitkeep
!persona_vectors/.gitkeep
!localization/.gitkeep

# Old v1 directories (should not exist)
persona_vectors/
localization/
eval_results/

# Cleanup marker
.cleanup_done
"""

    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            content = f.read()

        if "v2 canonical" not in content:
            with open(gitignore_path, "a") as f:
                f.write(gitignore_additions)
            log("  Updated .gitignore with v2 canonical rules")
        else:
            log("  .gitignore already updated")

    log("\n" + "=" * 60)
    log("CLEANUP COMPLETE")
    log("=" * 60)
    log("\nCurrent structure:")
    log("  activations/          - Activation files (canonical)")
    log("  persona_vectors/   - Persona vectors (canonical)")
    log("  localization/      - Localization results (canonical)")
    log("  steering_results/     - Steering outputs (canonical)")
    log("  cross_model_results/  - Cross-model analysis (canonical)")
    log("")
    log("Removed:")
    for d in dirs_to_remove:
        log(f"  {d}/")
    log("")
    log(f"Total space freed: {total_freed:.1f} MB")
    log("\n✓ Ready for clean execution!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
