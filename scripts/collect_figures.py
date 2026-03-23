import os
import shutil
from pathlib import Path


def collect_and_rename_figures():
    paper_figs_dir = Path("paper/figures")
    paper_figs_dir.mkdir(parents=True, exist_ok=True)

    import os

    if os.path.exists("results/persona_vectors"):
        models = [
            d
            for d in os.listdir("results/persona_vectors")
            if os.path.isdir(os.path.join("results/persona_vectors", d))
        ]
    else:
        models = []

    # 1. Layer Analysis (Encoding Profile)
    # Source: results/persona_vectors/{model}/{trait}/layer_analysis_v2_{trait}.png
    for model in models:
        src = Path(
            f"results/persona_vectors/{model}/openness/layer_analysis_v2_openness.png"
        )
        if src.exists():
            dst = paper_figs_dir / f"layer_profile_{model}_openness.png"
            shutil.copy(src, dst)
            print(f"Copied {dst.name}")

    # 2. Causal Localization
    # Source: results/localization/{model}/refined_{trait}.png
    for model in models:
        src = Path(f"results/localization/{model}/refined_openness.png")
        if src.exists():
            dst = paper_figs_dir / f"causal_loc_{model}_openness.png"
            shutil.copy(src, dst)
            print(f"Copied {dst.name}")

    # 3. Orthogonality (Cross Trait)
    # Source: results/persona_vectors/{model}/cross_trait_comparison_v2.png
    for model in models:
        src = Path(f"results/persona_vectors/{model}/cross_trait_comparison_v2.png")
        if src.exists():
            dst = paper_figs_dir / f"ortho_{model}.png"
            shutil.copy(src, dst)
            print(f"Copied {dst.name}")


if __name__ == "__main__":
    collect_and_rename_figures()
