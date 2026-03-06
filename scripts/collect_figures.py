import os
import shutil
from pathlib import Path

def collect_and_rename_figures():
    paper_figs_dir = Path("paper/figures")
    paper_figs_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        "Qwen_Qwen3-0.6B",
        "Qwen_Qwen2.5-0.5B-Instruct",
        "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
        "unsloth_Llama-3.2-1B-Instruct",
        "unsloth_gemma-2-2b-it"
    ]
    
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
