import os
import shutil
from pathlib import Path

def collect_and_rename_figures():
    paper_figs_dir = Path("paper/figures")
    paper_figs_dir.mkdir(parents=True, exist_ok=True)
    
    import os
    if os.path.exists("persona_vectors_v2"):
        models = [d for d in os.listdir("persona_vectors_v2") if os.path.isdir(os.path.join("persona_vectors_v2", d))]
    else:
        models = []
    
    # 1. Layer Analysis (Encoding Profile)
    # Source: persona_vectors_v2/{model}/{trait}/layer_analysis_v2_{trait}.png
    for model in models:
        # Just grab openness as representative for the layer profile
        src = Path(f"persona_vectors_v2/{model}/openness/layer_analysis_v2_openness.png")
        if src.exists():
            dst = paper_figs_dir / f"layer_profile_{model}_openness.png"
            shutil.copy(src, dst)
            print(f"Copied {dst.name}")
            
    # 2. Causal Localization
    # Source: localization_v2/{model}/refined_{trait}.png
    for model in models:
        src = Path(f"localization_v2/{model}/refined_openness.png")
        if src.exists():
            dst = paper_figs_dir / f"causal_loc_{model}_openness.png"
            shutil.copy(src, dst)
            print(f"Copied {dst.name}")
            
    # 3. Orthogonality (Cross Trait)
    # Source: persona_vectors_v2/{model}/cross_trait_comparison_v2.png (or eval scripts output)
    # Since eval_orthogonality script outputs might not be uniformly named across all models automatically,
    # we'll look for what exists.
    for model in models:
        src = Path(f"persona_vectors_v2/{model}/cross_trait_comparison_v2.png")
        if src.exists():
            dst = paper_figs_dir / f"ortho_{model}.png"
            shutil.copy(src, dst)
            print(f"Copied {dst.name}")

if __name__ == "__main__":
    collect_and_rename_figures()
