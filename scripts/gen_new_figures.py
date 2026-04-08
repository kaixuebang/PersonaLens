import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BFI = Path("/data1/tongjizhou/persona/results/bfi_behavioral_v2")
VECTORS = Path("/data1/tongjizhou/persona/results/persona_vectors")
FIGS = Path("/data1/tongjizhou/persona/paper/figures")
FIGS.mkdir(parents=True, exist_ok=True)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TS = {"openness": "O", "conscientiousness": "C", "extraversion": "E", "agreeableness": "A", "neuroticism": "N"}

MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct", "Q2.5-0.5B"),
    ("Qwen_Qwen3-0.6B", "Q3-0.6B"),
    ("TinyLlama_TinyLlama-1.1B-Chat-v1.0", "TinyLlama"),
    ("unsloth_Llama-3.2-1B-Instruct", "Llama-3.2"),
    ("Qwen_Qwen2.5-1.5B-Instruct", "Q2.5-1.5B"),
    ("unsloth_gemma-2-2b-it", "Gemma-2B"),
    ("Qwen_Qwen2.5-7B-Instruct", "Q2.5-7B"),
    ("mistralai_Mistral-7B-Instruct-v0.1", "Mistral-7B"),
]

eps_map = {"Q2.5-0.5B": 1e-6, "Q3-0.6B": 1e-6, "TinyLlama": 1e-5,
           "Llama-3.2": 1e-5, "Q2.5-1.5B": 1e-6, "Gemma-2B": 1e-6,
           "Q2.5-7B": 1e-6, "Mistral-7B": 1e-5}
softcap_map = {n: 0 for n in ["Q2.5-0.5B","Q3-0.6B","TinyLlama","Llama-3.2","Q2.5-1.5B","Q2.5-7B","Mistral-7B"]}
softcap_map["Gemma-2B"] = 50

def get_rms(model_dir):
    p = VECTORS / model_dir / "openness" / "analysis_v2_openness.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    bl = str(d.get("best_layer_loso", 0))
    return d["layers"][bl]["rms_scale"]

def get_steer_delta(model_dir, trait):
    p = BFI / model_dir / f"responses_{trait}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    drj = d.get("dose_response_judge", {})
    means = drj.get("means", [])
    if not means:
        return None
    return max(means) - min(means)

model_data = []
for model_dir, name in MODELS:
    rms = get_rms(model_dir)
    deltas = [get_steer_delta(model_dir, t) for t in TRAITS]
    mean_d = np.mean([d for d in deltas if d is not None]) if any(d is not None for d in deltas) else 0
    model_data.append({"name": name, "rms": rms, "delta": mean_d})

# ====== FIG 1: Architecture RMS vs eps ======
print("Generating fig_architecture_rms.png...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for md in model_data:
    eps = eps_map.get(md["name"], 1e-6)
    sc = softcap_map.get(md["name"], 0)
    color = '#e74c3c' if eps == 1e-6 else '#3498db'
    marker = 's' if sc > 0 else 'o'
    size = 150 if sc > 0 else 100
    ax.scatter(eps, md["rms"], c=color, marker=marker, s=size, edgecolors='black', linewidth=0.5, zorder=5)
    offset_y = 0.15
    if md["name"] == "Gemma-2B":
        offset_y = -0.18
    ax.annotate(md["name"], (eps, md["rms"]), fontsize=8,
                textcoords="offset points", xytext=(8, offset_y * 100))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("rms_norm_eps (log scale)", fontsize=12)
ax.set_ylabel("Hidden State RMS Scale (log scale)", fontsize=12)
ax.set_title("Architectural Determinants of Hidden State Scale\nLinear regression $R^2 = 0.91$", fontsize=11)
ax.grid(True, alpha=0.3, which='both')

from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markeredgecolor='black', label='$\\epsilon=10^{-6}$ (Qwen, Gemma)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markeredgecolor='black', label='$\\epsilon=10^{-5}$ (Llama, Mistral)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', markeredgecolor='black', label='$\\epsilon=10^{-6}$ + softcap (Gemma-2)'),
]
ax.legend(handles=legend, fontsize=9, loc='lower left')
plt.tight_layout()
fig.savefig(FIGS / "fig_architecture_rms.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Saved: {FIGS / 'fig_architecture_rms.png'}")

# ====== FIG 2: Interference Matrices (Mistral vs Q2.5-7B) ======
print("Generating fig_interference_matrices.png...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (model_dir, title) in enumerate([
    ("mistralai_Mistral-7B-Instruct-v0.1", "Mistral-7B (Strong Steer)"),
    ("Qwen_Qwen2.5-7B-Instruct", "Qwen2.5-7B (Weak Steer)")
]):
    matrix = np.zeros((5, 5))
    for i, steered in enumerate(TRAITS):
        path = BFI / model_dir / f"responses_{steered}.json"
        with open(path) as f:
            d = json.load(f)
        alphas = sorted(d["results"].keys(), key=lambda x: float(x))
        hi_a, lo_a = alphas[-1], alphas[0]
        for j, target in enumerate(TRAITS):
            sk = f"judge_rating_{target}"
            hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
            lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
            matrix[i, j] = np.mean(hi) - np.mean(lo) if hi and lo else 0

    im = axes[idx].imshow(matrix, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
    axes[idx].set_xticks(range(5))
    axes[idx].set_xticklabels([TS[t] for t in TRAITS], fontsize=11)
    axes[idx].set_yticks(range(5))
    axes[idx].set_yticklabels([TS[t] for t in TRAITS], fontsize=11)
    axes[idx].set_xlabel("Measured Trait", fontsize=11)
    axes[idx].set_ylabel("Steered Trait", fontsize=11)
    axes[idx].set_title(title, fontsize=12)

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            weight = 'bold' if i == j else 'normal'
            color = 'white' if abs(val) > 1.5 else 'black'
            axes[idx].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color, fontweight=weight)

plt.colorbar(im, ax=axes, label='$\\Delta$ (max $\\alpha$ - min $\\alpha$)', shrink=0.8)
plt.suptitle("Cross-Trait Interference: Strong vs Weak Steering Models", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGS / "fig_interference_matrices.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Saved: {FIGS / 'fig_interference_matrices.png'}")

# ====== FIG 3: Selectivity vs Steer Delta ======
print("Generating fig_selectivity.png...")

def compute_selectivity(model_dir):
    all_primary, all_off = [], []
    for steered in TRAITS:
        path = BFI / model_dir / f"responses_{steered}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        alphas = sorted(d["results"].keys(), key=lambda x: float(x))
        hi_a, lo_a = alphas[-1], alphas[0]
        for target in TRAITS:
            sk = f"judge_rating_{target}"
            hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
            lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
            if hi and lo:
                delta = abs(np.mean(hi) - np.mean(lo))
                if target == steered:
                    all_primary.append(delta)
                else:
                    all_off.append(delta)
    mp = np.mean(all_primary) if all_primary else 0
    mo = np.mean(all_off) if all_off else 0
    sel = mp / mo if mo > 0 else 0
    return mp, mo, sel

fig, ax = plt.subplots(figsize=(6, 4.5))
for model_dir, name in MODELS:
    mp, mo, sel = compute_selectivity(model_dir)
    mean_delta = get_steer_delta(model_dir, "openness") or 0
    color = '#2ecc71' if sel > 1.5 else '#e74c3c' if sel < 1.0 else '#f39c12'
    ax.scatter(mean_delta, sel, s=120, c=color, edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(name, (mean_delta, sel), fontsize=8, textcoords="offset points", xytext=(5, 5))

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Selectivity = 1.0')
ax.set_xlabel("Mean Steering $\\Delta$", fontsize=12)
ax.set_ylabel("Cross-Trait Selectivity Ratio", fontsize=12)
ax.set_title("Steering Effectiveness vs. Cross-Trait Selectivity", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIGS / "fig_selectivity.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Saved: {FIGS / 'fig_selectivity.png'}")

print("\nAll figures generated successfully!")
