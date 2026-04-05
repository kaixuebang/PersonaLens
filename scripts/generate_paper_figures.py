import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO = Path("/data1/tongjizhou/persona")
BFI_DIR = REPO / "results" / "bfi_behavioral_v2"
ADJ_DIR = REPO / "results" / "bfi_adjusted_alpha"
VECTORS_DIR = REPO / "results" / "persona_vectors"
FIGS_DIR = REPO / "paper" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_SHORT = {"openness": "O", "conscientiousness": "C", "extraversion": "E", "agreeableness": "A", "neuroticism": "N"}

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

def get_steering_metrics(json_path):
    with open(json_path) as f:
        data = json.load(f)
    drj = data.get("dose_response_judge", {})
    alphas = drj.get("alphas", [])
    means = drj.get("means", [])
    if not alphas or not means:
        return None, None
    delta = max(means) - min(means)
    r, _ = stats.spearmanr(alphas, means) if len(alphas) >= 3 else (None, None)
    return delta, r

def get_rms_scale(model_dir, trait="openness"):
    analysis_path = VECTORS_DIR / model_dir / trait / f"analysis_v2_{trait}.json"
    if not analysis_path.exists():
        return None
    with open(analysis_path) as f:
        data = json.load(f)
    best_layer = data.get("best_layer_loso", 0)
    layer_data = data["layers"].get(str(best_layer), {})
    for lid in sorted(data["layers"].keys(), key=lambda x: int(x)):
        ld = data["layers"][lid]
        if ld.get("loso_accuracy", 0) == data.get("best_loso_accuracy", 0):
            return ld.get("rms_scale", None)
    return layer_data.get("rms_scale", None)

def load_persona_vectors(model_dir, layer_id=None):
    vectors = {}
    for trait in TRAITS:
        analysis_path = VECTORS_DIR / model_dir / trait / f"analysis_v2_{trait}.json"
        if not analysis_path.exists():
            continue
        with open(analysis_path) as f:
            data = json.load(f)
        if layer_id is None:
            layer_id_trait = str(data.get("best_layer_loso", 0))
        else:
            layer_id_trait = str(layer_id)
        vec_path = VECTORS_DIR / model_dir / trait / "vectors" / f"mean_diff_layer_{layer_id_trait}.npy"
        if vec_path.exists():
            v = np.load(vec_path)
            vectors[trait] = v / (np.linalg.norm(v) + 1e-10)
    return vectors

print("Collecting data...")

model_data = {}
for model_dir, model_name in MODELS:
    rms = get_rms_scale(model_dir)
    orig_deltas, orig_rs = [], []
    adj_deltas, adj_rs = [], []
    per_trait_orig = {}
    per_trait_adj = {}

    for trait in TRAITS:
        orig_path = BFI_DIR / model_dir / f"responses_{trait}.json"
        d, r = get_steering_metrics(orig_path) if orig_path.exists() else (None, None)
        per_trait_orig[trait] = (d, r)
        if d is not None:
            orig_deltas.append(d)
            orig_rs.append(r)

        adj_path = ADJ_DIR / model_dir / f"responses_{trait}.json"
        d, r = get_steering_metrics(adj_path) if adj_path.exists() else (None, None)
        per_trait_adj[trait] = (d, r)
        if d is not None:
            adj_deltas.append(d)
            adj_rs.append(r)

    vectors = load_persona_vectors(model_dir)
    cos_matrix = np.zeros((len(TRAITS), len(TRAITS)))
    for i, t1 in enumerate(TRAITS):
        for j, t2 in enumerate(TRAITS):
            if t1 in vectors and t2 in vectors:
                cos_matrix[i, j] = np.dot(vectors[t1], vectors[t2])

    off_diag = cos_matrix[np.triu_indices(len(TRAITS), k=1)]
    mean_abs_cos = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
    cond_number = np.linalg.cond(cos_matrix) if cos_matrix.size > 0 else 0

    model_data[model_name] = {
        "rms": rms,
        "orig_delta": np.mean(orig_deltas) if orig_deltas else 0,
        "orig_r": np.mean(orig_rs) if orig_rs else 0,
        "adj_delta": np.mean(adj_deltas) if adj_deltas else None,
        "adj_r": np.mean(adj_rs) if adj_rs else None,
        "per_trait_orig": per_trait_orig,
        "per_trait_adj": per_trait_adj,
        "mean_abs_cos": mean_abs_cos,
        "cond_number": cond_number,
        "cos_matrix": cos_matrix,
    }
    print(f"  {model_name}: RMS={rms:.3f}, orig_Δ={model_data[model_name]['orig_delta']:.2f}, "
          f"entangle(|cos|)={mean_abs_cos:.3f}, cond={cond_number:.1f}")


# ========================================================================
# FIGURE 1: PR Scatter Plot (RMS scale vs Steering Δ)
# ========================================================================
print("\nGenerating PR scatter plot...")
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

model_names_ordered = [n for _, n in MODELS]
rms_vals = [model_data[m]["rms"] for m in model_names_ordered if model_data[m]["rms"] is not None]
delta_vals = [model_data[m]["orig_delta"] for m in model_names_ordered if model_data[m]["rms"] is not None]
names_clean = [m for m in model_names_ordered if model_data[m]["rms"] is not None]
rms_clean = rms_vals
delta_clean = delta_vals

r_val, p_val = stats.spearmanr(rms_clean, delta_clean)

colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(rms_clean)))
sort_idx = np.argsort(delta_clean)[::-1]
color_map = {names_clean[i]: colors[j] for j, i in enumerate(sort_idx)}

for name, rms, delta in zip(names_clean, rms_clean, delta_clean):
    ax.scatter(rms, delta, s=120, c=[color_map[name]], edgecolors='black', linewidth=0.5, zorder=5)
    offset_x, offset_y = 5, 5
    if name == "Q2.5-0.5B":
        offset_y = -12
    elif name == "Q2.5-7B":
        offset_y = -12
    elif name == "Gemma-2B":
        offset_x = -15
        offset_y = -12
    ax.annotate(name, (rms, delta), fontsize=8, textcoords="offset points",
                xytext=(offset_x, offset_y), ha='left')

ax.set_xlabel("RMS Scale (hidden state magnitude)", fontsize=11)
ax.set_ylabel("Steering Effectiveness Δ", fontsize=11)
ax.set_title(f"Hidden State Scale vs. Steering Effectiveness ($\\rho = {r_val:.3f}$, $p = {p_val:.3f}$)", fontsize=11)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIGS_DIR / "fig_pr_scatter.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig_pr_scatter.png saved")


# ========================================================================
# FIGURE 2: Steering Heatmap (8 models × 5 traits)
# ========================================================================
print("Generating steering heatmap...")
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

model_order = [n for _, n in MODELS]
heatmap_data = np.zeros((len(model_order), len(TRAITS)))
for i, name in enumerate(model_order):
    for j, trait in enumerate(TRAITS):
        d, _ = model_data[name]["per_trait_orig"].get(trait, (0, None))
        heatmap_data[i, j] = d if d is not None else 0

im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2.7)
ax.set_xticks(range(len(TRAITS)))
ax.set_xticklabels([TRAIT_SHORT[t] for t in TRAITS], fontsize=11)
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=10)

for i in range(len(model_order)):
    for j in range(len(TRAITS)):
        val = heatmap_data[i, j]
        color = 'white' if val > 1.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

cbar = plt.colorbar(im, ax=ax, label='Steering Δ')
ax.set_title("Cross-Model Steering Effectiveness by Trait", fontsize=12)
plt.tight_layout()
fig.savefig(FIGS_DIR / "fig_steering_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig_steering_heatmap.png saved")


# ========================================================================
# FIGURE 3: Adjusted Alpha Comparison Bar Chart
# ========================================================================
print("Generating adjusted alpha comparison...")
adj_models = ["Gemma-2B", "Q3-0.6B", "Q2.5-1.5B", "Q2.5-7B"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

orig_ds = [model_data[m]["orig_delta"] for m in adj_models]
adj_ds = [model_data[m]["adj_delta"] for m in adj_models]
orig_rs = [model_data[m]["orig_r"] for m in adj_models]
adj_rs = [model_data[m]["adj_r"] for m in adj_models]

x = np.arange(len(adj_models))
width = 0.35

bars1 = axes[0].bar(x - width/2, orig_ds, width, label='Original α', color='#e74c3c', alpha=0.8)
bars2 = axes[0].bar(x + width/2, adj_ds, width, label='Adjusted α', color='#2ecc71', alpha=0.8)
axes[0].set_ylabel('Mean Steering Δ')
axes[0].set_title('Steering Effectiveness')
axes[0].set_xticks(x)
axes[0].set_xticklabels(adj_models, fontsize=9)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8)

bars3 = axes[1].bar(x - width/2, orig_rs, width, label='Original α', color='#e74c3c', alpha=0.8)
bars4 = axes[1].bar(x + width/2, adj_rs, width, label='Adjusted α', color='#2ecc71', alpha=0.8)
axes[1].set_ylabel('Mean Monotonicity r (Spearman)')
axes[1].set_title('Dose-Response Monotonicity')
axes[1].set_xticks(x)
axes[1].set_xticklabels(adj_models, fontsize=9)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0, color='black', linewidth=0.5)

for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars4:
    y_pos = bar.get_height() + 0.03 if bar.get_height() >= 0 else bar.get_height() - 0.08
    axes[1].text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig_adjusted_alpha_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig_adjusted_alpha_comparison.png saved")


# ========================================================================
# FIGURE 4: Entanglement Analysis
# ========================================================================
print("Generating entanglement analysis...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

model_names_list = [n for _, n in MODELS]
entangle_vals = [model_data[m]["mean_abs_cos"] for m in model_names_list]
steer_vals = [model_data[m]["orig_delta"] for m in model_names_list]
adj_steer_vals = [model_data[m]["adj_delta"] if model_data[m]["adj_delta"] is not None
                  else model_data[m]["orig_delta"] for m in model_names_list]

axes[0].scatter(entangle_vals, steer_vals, s=100, edgecolors='black', linewidth=0.5, c='#3498db')
for name, ev, sv in zip(model_names_list, entangle_vals, steer_vals):
    axes[0].annotate(name, (ev, sv), fontsize=7, textcoords="offset points", xytext=(5, 5))
r_ent, p_ent = stats.spearmanr(entangle_vals, steer_vals)
axes[0].set_xlabel("Mean |cos| between persona vectors", fontsize=10)
axes[0].set_ylabel("Original Steering Δ", fontsize=10)
axes[0].set_title(f"Trait Entanglement vs. Steering ($\\rho={r_ent:.3f}$, $p={p_ent:.3f}$)", fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(entangle_vals, adj_steer_vals, s=100, edgecolors='black', linewidth=0.5, c='#2ecc71')
for name, ev, sv in zip(model_names_list, entangle_vals, adj_steer_vals):
    axes[1].annotate(name, (ev, sv), fontsize=7, textcoords="offset points", xytext=(5, 5))
r_ent2, p_ent2 = stats.spearmanr(entangle_vals, adj_steer_vals)
axes[1].set_xlabel("Mean |cos| between persona vectors", fontsize=10)
axes[1].set_ylabel("Adjusted Steering Δ (best available)", fontsize=10)
axes[1].set_title(f"Trait Entanglement vs. Adj. Steering ($\\rho={r_ent2:.3f}$, $p={p_ent2:.3f}$)", fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig_entanglement_analysis.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig_entanglement_analysis.png saved")


# ========================================================================
# Print corrected per-trait tables for paper update
# ========================================================================
print("\n" + "=" * 80)
print("CORRECTED PER-TRAIT TABLES FOR PAPER")
print("=" * 80)

for model_dir, model_name in MODELS:
    adj_path_prefix = ADJ_DIR / model_dir
    has_adj = (adj_path_prefix / "responses_openness.json").exists()
    if not has_adj:
        continue
    
    with open(adj_path_prefix / "responses_openness.json") as f:
        d = json.load(f)
    adj_alphas = d.get("alphas", [])
    
    scale_map = {
        "unsloth_gemma-2-2b-it": 23,
        "Qwen_Qwen3-0.6B": 9,
        "Qwen_Qwen2.5-1.5B-Instruct": 5,
        "Qwen_Qwen2.5-7B-Instruct": 5,
    }
    scale = scale_map.get(model_dir, "?")
    
    print(f"\n% {model_name} (scale={scale}x, alphas={adj_alphas})")
    print(f"Trait & Orig $\\Delta$ & Adj $\\Delta$ & Orig $r$ & Adj $r$ \\\\")
    for trait in TRAITS:
        od, orv = model_data[model_name]["per_trait_orig"].get(trait, (None, None))
        ad, arv = model_data[model_name]["per_trait_adj"].get(trait, (None, None))
        od_s = f"{od:.2f}" if od is not None else "---"
        or_s = f"{orv:.2f}" if orv is not None else "---"
        ad_s = f"{ad:.2f}" if ad is not None else "---"
        ar_s = f"{arv:.2f}" if arv is not None else "---"
        print(f"{trait.capitalize()} & {od_s} & {ad_s} & {or_s} & {ar_s} \\\\")

print("\n\nDone!")
