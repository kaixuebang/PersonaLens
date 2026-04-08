import json
import numpy as np
from pathlib import Path

BFI_ORIG = Path("results/bfi_behavioral_v2")
BFI_ADJ = Path("results/bfi_adjusted_alpha")
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

ADJ_MODELS = [
    ("Qwen_Qwen2.5-1.5B-Instruct", "Q2.5-1.5B"),
    ("Qwen_Qwen2.5-7B-Instruct", "Q2.5-7B"),
    ("Qwen_Qwen3-0.6B", "Q3-0.6B"),
    ("unsloth_gemma-2-2b-it", "Gemma-2B"),
]


def compute_selectivity(source_dir, model_dir_name):
    all_primary = []
    all_off = []

    for steered in TRAITS:
        path = source_dir / model_dir_name / f"responses_{steered}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)

        alpha_keys = sorted(d["results"].keys(), key=lambda x: float(x))
        hi_a = alpha_keys[-1]
        lo_a = alpha_keys[0]

        for target in TRAITS:
            sk = f"judge_rating_{target}"
            hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
            lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
            if not hi or not lo:
                continue
            delta = abs(np.mean(hi) - np.mean(lo))
            if target == steered:
                all_primary.append(delta)
            else:
                all_off.append(delta)

    mp = np.mean(all_primary) if all_primary else 0.0
    mo = np.mean(all_off) if all_off else 0.001
    mx = max(all_off) if all_off else 0.0
    sel = mp / mo
    return mp, mo, mx, sel


print("=" * 90)
print("ORIGINAL vs ADJUSTED-ALPHA SELECTIVITY COMPARISON")
print("=" * 90)

results = {}

for model_dir, name in ADJ_MODELS:
    orig_p, orig_o, orig_x, orig_sel = compute_selectivity(BFI_ORIG, model_dir)
    adj_p, adj_o, adj_x, adj_sel = compute_selectivity(BFI_ADJ, model_dir)

    results[name] = {
        "orig_primary": round(orig_p, 2),
        "orig_off": round(orig_o, 2),
        "orig_sel": round(orig_sel, 1),
        "adj_primary": round(adj_p, 2),
        "adj_off": round(adj_o, 2),
        "adj_sel": round(adj_sel, 1),
    }

print(f"\n{'Model':<12} | {'Orig Δ':>7} {'Off':>5} {'Sel':>5} | {'Adj Δ':>7} {'Off':>5} {'Sel':>5} | {'ΔSel':>6}")
print("-" * 75)

for name in [n for _, n in ADJ_MODELS]:
    r = results[name]
    dsel = r["adj_sel"] - r["orig_sel"]
    sign = "+" if dsel >= 0 else ""
    print(f"{name:<12} | {r['orig_primary']:>7.2f} {r['orig_off']:>5.2f} {r['orig_sel']:>4.1f}x | "
          f"{r['adj_primary']:>7.2f} {r['adj_off']:>5.2f} {r['adj_sel']:>4.1f}x | {sign}{dsel:>5.1f}x")

with open("results/cross_trait_interference/adjusted_selectivity_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/cross_trait_interference/adjusted_selectivity_comparison.json")
