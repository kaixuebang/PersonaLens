"""
Regenerate ALL main-text and key appendix figures with 14 models.
"""
import json, numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path("/data1/tongjizhou/persona")
BFI_DIR = REPO / "results" / "bfi_behavioral_v2"
VECTORS_DIR = REPO / "results" / "persona_vectors"
ENT_DIR = REPO / "results" / "entanglement_metrics"
ADJ_DIR = REPO / "results" / "bfi_adjusted_alpha"
FIGS = REPO / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TS = {"openness": "O", "conscientiousness": "C", "extraversion": "E", "agreeableness": "A", "neuroticism": "N"}

# All 14 models sorted by parameter count (ascending)
ALL_MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct",                    "Q2.5-0.5B",      0.5,  "qwen2",  1e-6, 24,  896),
    ("Qwen_Qwen3-0.6B",                                "Q3-0.6B",        0.6,  "qwen2",  1e-6, 28, 1024),
    ("TinyLlama_TinyLlama-1.1B-Chat-v1.0",             "TinyLlama",      1.1,  "llama",  1e-5, 22, 2048),
    ("unsloth_Llama-3.2-1B-Instruct",                   "Llama-3.2",      1.2,  "llama",  1e-5, 16, 2048),
    ("Qwen_Qwen2.5-1.5B-Instruct",                     "Q2.5-1.5B",      1.5,  "qwen2",  1e-6, 28, 1536),
    ("unsloth_gemma-2-2b-it",                           "Gemma-2B",       2.6,  "gemma2", 1e-6, 26, 2304),
    ("_data0_shizitong_models_Phi3-mini-128k-instruct", "Phi-3-mini",     3.8,  "phi3",   1e-5, 32, 3072),
    ("Qwen_Qwen2.5-7B-Instruct",                        "Q2.5-7B",        7.6,  "qwen2",  1e-6, 28, 3584),
    ("mistralai_Mistral-7B-Instruct-v0.1",              "Mistral-7B",     7.2,  "mistral",1e-5, 32, 4096),
    ("unsloth_llama-3-8B-Instruct",                      "Llama-3-8B",     8.0,  "llama",  1e-5, 32, 4096),
    ("unsloth_Llama-3.1-8B-Instruct",                    "Llama-3.1-8B",   8.0,  "llama",  1e-5, 32, 4096),
    ("_data0_shizitong_models_Llama-2-7b-chat-hf",      "Llama-2-7B",     6.7,  "llama",  1e-5, 32, 4096),
    ("_data0_shizitong_models_DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-14B", 14.7, "qwen2", 1e-5, 48, 5120),
    ("Qwen_Qwen2.5-14B-Instruct",                       "Q2.5-14B",      14.3,  "qwen2",  1e-6, 48, 5120),
]

def get_rms(model_dir, trait="openness"):
    p = VECTORS_DIR / model_dir / trait / f"analysis_v2_{trait}.json"
    if not p.exists(): return None
    d = json.load(open(p))
    for lid in sorted(d["layers"].keys(), key=lambda x: int(x)):
        ld = d["layers"][lid]
        if ld.get("loso_accuracy", 0) == d.get("best_loso_accuracy", 0):
            return ld.get("rms_scale", None)
    return None

def get_steer_metrics(model_dir, trait):
    p = BFI_DIR / model_dir / f"responses_{trait}.json"
    if not p.exists(): return None, None, None, None
    d = json.load(open(p))
    alphas = sorted([float(k) for k in d["results"].keys()])
    means = []
    for a in alphas:
        ratings = [r['judge_rating'] for r in d["results"][str(float(a))]["scenario_results"] if 'judge_rating' in r]
        means.append(np.mean(ratings) if ratings else 0)
    delta = max(means) - min(means)
    rho, _ = stats.spearmanr(alphas, means) if len(alphas) >= 3 else (None, None)
    return delta, rho, alphas, means

def get_entanglement(model_dir):
    p = ENT_DIR / f"{model_dir}_entanglement.json"
    if not p.exists(): return None
    return json.load(open(p))

def load_vectors(model_dir):
    vectors = {}
    for trait in TRAITS:
        ap = VECTORS_DIR / model_dir / trait / f"analysis_v2_{trait}.json"
        if not ap.exists(): continue
        d = json.load(open(ap))
        bl = str(d.get("best_layer_loso", 0))
        vp = VECTORS_DIR / model_dir / trait / "vectors" / f"mean_diff_layer_{bl}.npy"
        if vp.exists():
            v = np.load(vp)
            vectors[trait] = v / (np.linalg.norm(v) + 1e-10)
    return vectors

# ========== Collect all data ==========
print("Collecting data for 14 models...")
data = {}
for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    rms = get_rms(model_dir)
    deltas, rhos, trait_data = [], [], {}
    for t in TRAITS:
        d, r, als, ms = get_steer_metrics(model_dir, t)
        trait_data[t] = {"delta": d, "rho": r, "alphas": als, "means": ms}
        if d is not None: deltas.append(d)
        if r is not None: rhos.append(r)
    
    vectors = load_vectors(model_dir)
    cos_mat = np.zeros((5,5))
    for i,t1 in enumerate(TRAITS):
        for j,t2 in enumerate(TRAITS):
            if t1 in vectors and t2 in vectors:
                cos_mat[i,j] = np.dot(vectors[t1], vectors[t2])
    off_diag = cos_mat[np.triu_indices(5, k=1)]
    
    ent = get_entanglement(model_dir)
    
    data[name] = {
        "dir": model_dir, "params": params, "arch": arch, "eps": eps,
        "nlayers": nlayers, "hdim": hdim, "rms": rms,
        "mean_delta": np.mean(deltas) if deltas else 0,
        "mean_rho": np.mean(rhos) if rhos else 0,
        "trait_data": trait_data,
        "cos_matrix": cos_mat,
        "mean_abs_cos": np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0,
        "ent": ent,
    }
    sig_count = sum(1 for t in TRAITS if trait_data[t]["rho"] is not None and abs(trait_data[t]["rho"]) > 0.5)
    rms_str = f"{rms:.3f}" if rms else "N/A"
    print(f"  {name:20s} | Δ={data[name]['mean_delta']:.3f} | ρ={data[name]['mean_rho']:.3f} | RMS={rms_str:>6s} | sig={sig_count}/5")

# Helper for coloring by epsilon
def eps_color(eps):
    return '#e74c3c' if eps == 1e-6 else '#3498db'

def arch_marker(arch):
    return {'qwen2': 'o', 'llama': 's', 'mistral': '^', 'gemma2': 'D', 'phi3': 'v'}.get(arch, 'o')

# ========================================================================
# FIGURE 1: RMS vs Delta scatter (main text)
# ========================================================================
print("\n[1/7] Generating fig_rms_vs_delta.png (14 models)...")
fig, ax = plt.subplots(figsize=(7, 5))

rms_all = [data[n]["rms"] for n in [m[1] for m in ALL_MODELS] if data[n]["rms"] is not None]
delta_all = [data[n]["mean_delta"] for n in [m[1] for m in ALL_MODELS] if data[n]["rms"] is not None]
names_all = [n for n in [m[1] for m in ALL_MODELS] if data[n]["rms"] is not None]

r_val, p_val = stats.spearmanr(rms_all, delta_all)

for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    rms = data[name]["rms"]
    delta = data[name]["mean_delta"]
    if rms is None: continue
    c = eps_color(eps)
    m = arch_marker(arch)
    ax.scatter(rms, delta, c=c, marker=m, s=120+params*5, edgecolors='black', linewidth=0.5, zorder=5)
    # Smart label placement
    ox, oy = 6, 6
    if name == "Gemma-2B": ox, oy = -40, -12
    elif name == "Q2.5-0.5B": ox, oy = 6, -10
    elif name == "Q3-0.6B": ox, oy = 6, -10
    elif name == "Q2.5-1.5B": ox, oy = -50, -10
    elif name == "Llama-2-7B": ox, oy = 6, 8
    elif name == "Phi-3-mini": ox, oy = 6, 6
    elif name == "Mistral-7B": ox, oy = -60, 8
    elif name == "Q2.5-7B": ox, oy = 6, -10
    elif name == "Llama-3-8B": ox, oy = 6, 8
    elif name == "Llama-3.1-8B": ox, oy = -70, -12
    elif name == "Q2.5-14B": ox, oy = 6, 6
    elif name == "DeepSeek-R1-14B": ox, oy = -90, -8
    elif name == "TinyLlama": ox, oy = 6, 6
    elif name == "Llama-3.2": ox, oy = 6, 6
    ax.annotate(name, (rms, delta), fontsize=7, textcoords="offset points", xytext=(ox, oy), ha='left')

# Add correlation line
log_rms = np.log10(rms_all)
z = np.polyfit(log_rms, delta_all, 1)
p_line = np.poly1d(z)
x_fit = np.linspace(min(log_rms)-0.1, max(log_rms)+0.1, 50)
ax.plot(10**x_fit, p_line(x_fit), '--', color='gray', alpha=0.5, linewidth=1)

ax.set_xscale('log')
ax.set_xlabel("Hidden State RMS Scale (log)", fontsize=12)
ax.set_ylabel("Mean Steering Effectiveness $\\Delta$", fontsize=12)
ax.set_title(f"RMS Scale vs. Steering Effectiveness ($\\rho = {r_val:.3f}$, $p = {p_val:.4f}$)", fontsize=12)
ax.grid(True, alpha=0.3)

legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=eps_color(1e-6), markeredgecolor='black', label='$\\epsilon=10^{-6}$ (Qwen, Gemma)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=eps_color(1e-5), markeredgecolor='black', label='$\\epsilon=10^{-5}$ (Llama, Mistral, Phi, DeepSeek)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
plt.tight_layout()
fig.savefig(FIGS / "fig_rms_vs_delta.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_rms_vs_delta.png")

# ========================================================================
# FIGURE 2: Steering curves (main text) - representative models
# ========================================================================
print("\n[2/7] Generating fig_steering_curves.png (14 models)...")
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
rep_models = [
    ("Llama-3-8B", "Strong ($\\epsilon=10^{-5}$)"),
    ("Mistral-7B", "Strong ($\\epsilon=10^{-5}$)"),
    ("Q2.5-0.5B", "Weak ($\\epsilon=10^{-6}$)"),
    ("Gemma-2B", "Weak ($\\epsilon=10^{-6}$)"),
    ("Phi-3-mini", "Anomaly ($\\epsilon=10^{-5}$)"),
    ("DeepSeek-R1-14B", "Anomaly ($\\epsilon=10^{-5}$)"),
    ("Llama-2-7B", "Anomaly ($\\epsilon=10^{-5}$)"),
    ("Q2.5-14B", "Weak ($\\epsilon=10^{-6}$)"),
]

trait_colors = {'openness': '#e74c3c', 'conscientiousness': '#3498db', 'extraversion': '#2ecc71',
                'agreeableness': '#f39c12', 'neuroticism': '#9b59b6'}

for idx, (name, subtitle) in enumerate(rep_models):
    ax = axes[idx // 4, idx % 4]
    td = data[name]["trait_data"]
    for t in TRAITS:
        if td[t]["alphas"] is None: continue
        als = td[t]["alphas"]
        ms = td[t]["means"]
        ax.plot(als, ms, '-o', markersize=3, color=trait_colors[t], label=TS[t] if idx == 0 else None, alpha=0.8)
    ax.set_title(f"{name}\n{subtitle}", fontsize=9)
    ax.set_xlabel("$\\alpha$", fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel("Judge Rating", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 5.5)

fig.legend([plt.Line2D([0],[0], color=trait_colors[t], marker='o', markersize=5, linestyle='-') for t in TRAITS],
           [TS[t] for t in TRAITS], loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Dose-Response Curves Across Representative Models", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGS / "fig_steering_curves.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_steering_curves.png")

# ========================================================================
# FIGURE 3: Steering heatmap (14 models x 5 traits)
# ========================================================================
print("\n[3/7] Generating fig_steering_heatmap.png (14 models)...")
# Order by mean delta descending
model_order = sorted([m[1] for m in ALL_MODELS], key=lambda n: data[n]["mean_delta"], reverse=True)

fig, ax = plt.subplots(figsize=(8, 7))
heatmap = np.zeros((len(model_order), 5))
for i, name in enumerate(model_order):
    for j, t in enumerate(TRAITS):
        d = data[name]["trait_data"][t]["delta"]
        heatmap[i, j] = d if d is not None else 0

im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3.0)
ax.set_xticks(range(5))
ax.set_xticklabels([TS[t] for t in TRAITS], fontsize=11)
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=9)

for i in range(len(model_order)):
    for j in range(5):
        val = heatmap[i,j]
        color = 'white' if val > 1.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

cbar = plt.colorbar(im, ax=ax, label='Steering $\\Delta$')
ax.set_title("Cross-Model Steering Effectiveness by Trait (14 Models)", fontsize=12)
plt.tight_layout()
fig.savefig(FIGS / "fig_steering_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_steering_heatmap.png")

# ========================================================================
# FIGURE 4: Selectivity vs Delta
# ========================================================================
print("\n[4/7] Generating fig_selectivity.png (14 models)...")
fig, ax = plt.subplots(figsize=(7, 5))

for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    td = data[name]["trait_data"]
    primary, off_diag = [], []
    for t_steer in TRAITS:
        for t_target in TRAITS:
            d_target = td[t_steer]["delta"]  # we need cross-trait delta
    td = data[name]["trait_data"]
    primary_deltas, off_diag_deltas = [], []
    for t_steer in TRAITS:
        pass
    ent = data[name]["ent"]
    if ent and "aggregate_entanglement" in ent:
        ae = ent["aggregate_entanglement"]
        sel = ae.get("selectivity", 0)
    elif ent and "cross_probe" in ent:
        cross = ent["cross_probe"]["mean_cross_acc"]
        rv = ent.get("rv", {}).get("mean", 1.0)
        sel = (1 - rv) / cross if cross > 0 else 0
    else:
        sel = 0
    
    c = eps_color(eps)
    m = arch_marker(arch)
    ax.scatter(data[name]["mean_delta"], sel, c=c, marker=m, s=120, edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(name, (data[name]["mean_delta"], sel), fontsize=7, textcoords="offset points", xytext=(5, 5))

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Selectivity = 1.0')
ax.set_xlabel("Mean Steering $\\Delta$", fontsize=12)
ax.set_ylabel("Selectivity (Independent / Cross-probe)", fontsize=12)
ax.set_title("Steering Effectiveness vs. Cross-Trait Selectivity (14 Models)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIGS / "fig_selectivity.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_selectivity.png")

# ========================================================================
# FIGURE 5: Architecture RMS plot (appendix)
# ========================================================================
print("\n[5/7] Generating fig_architecture_rms.png (14 models)...")
fig, ax = plt.subplots(figsize=(8, 5))

for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    rms = data[name]["rms"]
    if rms is None: continue
    c = eps_color(eps)
    m = arch_marker(arch)
    size = 80 + params * 8
    ax.scatter(eps, rms, c=c, marker=m, s=size, edgecolors='black', linewidth=0.5, zorder=5)
    ox, oy = 8, 0.15
    if name in ["Llama-3-8B", "Llama-3.1-8B", "Mistral-7B", "Llama-2-7B"]:
        oy = -0.2
        if name == "Llama-3.1-8B": ox = -50
        if name == "Llama-2-7B": ox = 8; oy = 0.18
    elif name == "Gemma-2B": oy = -0.18
    elif name == "Q2.5-1.5B": oy = -0.15
    ax.annotate(name, (eps, rms), fontsize=7, textcoords="offset points", xytext=(ox, oy*100))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("rms_norm_eps (log scale)", fontsize=12)
ax.set_ylabel("Hidden State RMS Scale (log scale)", fontsize=12)
ax.set_title("Architectural Determinants of Hidden State Scale (14 Models)", fontsize=11)
ax.grid(True, alpha=0.3, which='both')

legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=eps_color(1e-6), markeredgecolor='black', markersize=8, label='$\\epsilon=10^{-6}$ (Qwen, Gemma)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=eps_color(1e-5), markeredgecolor='black', markersize=8, label='$\\epsilon=10^{-5}$ (Llama, Mistral)'),
    Line2D([0],[0], marker='v', color='w', markerfacecolor=eps_color(1e-5), markeredgecolor='black', markersize=8, label='$\\epsilon=10^{-5}$ (Phi, DeepSeek)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower left')
plt.tight_layout()
fig.savefig(FIGS / "fig_architecture_rms.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_architecture_rms.png")

# ========================================================================
# FIGURE 6: Interference matrices - expand to include 3 new models
# ========================================================================
print("\n[6/7] Generating fig_interference_matrices.png...")
# Show 4 representative models: Strong, Weak, and 2 anomalies
int_models = [
    ("mistralai_Mistral-7B-Instruct-v0.1", "Mistral-7B\n(Strong, $\\epsilon=10^{-5}$)"),
    ("Qwen_Qwen2.5-7B-Instruct", "Q2.5-7B\n(Weak, $\\epsilon=10^{-6}$)"),
    ("_data0_shizitong_models_Phi3-mini-128k-instruct", "Phi-3-mini\n(Anomaly, $\\epsilon=10^{-5}$)"),
    ("_data0_shizitong_models_DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-14B\n(Anomaly, $\\epsilon=10^{-5}$)"),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for idx, (model_dir, title) in enumerate(int_models):
    matrix = np.zeros((5, 5))
    for i, steered in enumerate(TRAITS):
        p = BFI_DIR / model_dir / f"responses_{steered}.json"
        d = json.load(open(p))
        alphas = sorted(d["results"].keys(), key=lambda x: float(x))
        hi_a, lo_a = alphas[-1], alphas[0]
        for j, target in enumerate(TRAITS):
            sk = f"judge_rating_{target}"
            hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
            lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
            matrix[i,j] = np.mean(hi) - np.mean(lo) if hi and lo else 0
    
    im = axes[idx].imshow(matrix, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
    axes[idx].set_xticks(range(5))
    axes[idx].set_xticklabels([TS[t] for t in TRAITS], fontsize=9)
    axes[idx].set_yticks(range(5))
    axes[idx].set_yticklabels([TS[t] for t in TRAITS], fontsize=9)
    axes[idx].set_xlabel("Measured", fontsize=9)
    if idx == 0: axes[idx].set_ylabel("Steered", fontsize=9)
    axes[idx].set_title(title, fontsize=9)
    
    for i in range(5):
        for j in range(5):
            val = matrix[i,j]
            weight = 'bold' if i==j else 'normal'
            color = 'white' if abs(val) > 1.5 else 'black'
            axes[idx].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight=weight)

plt.colorbar(im, ax=axes, label='$\\Delta$ (max $\\alpha$ $-$ min $\\alpha$)', shrink=0.8)
plt.suptitle("Cross-Trait Interference Across Model Categories", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGS / "fig_interference_matrices.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_interference_matrices.png")

# ========================================================================
# FIGURE 7: Entanglement analysis (appendix)
# ========================================================================
print("\n[7/7] Generating fig_entanglement_analysis.png (14 models)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

names_list = [m[1] for m in ALL_MODELS]
ent_vals = [data[n]["mean_abs_cos"] for n in names_list]
steer_vals = [data[n]["mean_delta"] for n in names_list]

for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    c = eps_color(eps)
    m = arch_marker(arch)
    axes[0].scatter(data[name]["mean_abs_cos"], data[name]["mean_delta"],
                   c=c, marker=m, s=100, edgecolors='black', linewidth=0.5, zorder=5)
    axes[0].annotate(name, (data[name]["mean_abs_cos"], data[name]["mean_delta"]),
                    fontsize=6, textcoords="offset points", xytext=(5,5))

r_ent, p_ent = stats.spearmanr(ent_vals, steer_vals)
axes[0].set_xlabel("Mean |cos| between persona vectors", fontsize=10)
axes[0].set_ylabel("Original Steering $\\Delta$", fontsize=10)
axes[0].set_title(f"Trait Entanglement vs. Steering ($\\rho={r_ent:.3f}$, $p={p_ent:.3f}$)", fontsize=10)
axes[0].grid(True, alpha=0.3)

# Right panel: entanglement vs rho
rho_vals = [data[n]["mean_rho"] for n in names_list]
for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    c = eps_color(eps)
    m = arch_marker(arch)
    axes[1].scatter(data[name]["mean_abs_cos"], abs(data[name]["mean_rho"]),
                   c=c, marker=m, s=100, edgecolors='black', linewidth=0.5, zorder=5)
    axes[1].annotate(name, (data[name]["mean_abs_cos"], abs(data[name]["mean_rho"])),
                    fontsize=6, textcoords="offset points", xytext=(5,5))

r_ent2, p_ent2 = stats.spearmanr(ent_vals, [abs(r) for r in rho_vals])
axes[1].set_xlabel("Mean |cos| between persona vectors", fontsize=10)
axes[1].set_ylabel("Mean |$\\rho$| (monotonicity)", fontsize=10)
axes[1].set_title(f"Trait Entanglement vs. Monotonicity ($\\rho={r_ent2:.3f}$, $p={p_ent2:.3f}$)", fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIGS / "fig_entanglement_analysis.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_entanglement_analysis.png")

# ========================================================================
# Bonus: PR scatter (appendix)
# ========================================================================
print("\n[bonus] Generating fig_pr_scatter.png (14 models)...")
fig, ax = plt.subplots(figsize=(7, 5))

rms_vals = [data[n]["rms"] for n in names_list if data[n]["rms"] is not None]
delta_vals = [data[n]["mean_delta"] for n in names_list if data[n]["rms"] is not None]
pr_names = [n for n in names_list if data[n]["rms"] is not None]
r_val, p_val = stats.spearmanr(rms_vals, delta_vals)

for model_dir, name, params, arch, eps, nlayers, hdim in ALL_MODELS:
    rms = data[name]["rms"]
    if rms is None: continue
    c = eps_color(eps)
    m = arch_marker(arch)
    ax.scatter(rms, data[name]["mean_delta"], c=c, marker=m, s=120+params*5,
              edgecolors='black', linewidth=0.5, zorder=5)
    ox, oy = 5, 5
    if name in ["Gemma-2B"]: ox, oy = -45, -10
    elif name in ["Q2.5-0.5B", "Q2.5-1.5B"]: oy = -10
    elif name == "Llama-3.1-8B": ox, oy = -65, -10
    elif name == "DeepSeek-R1-14B": ox, oy = -85, -8
    ax.annotate(name, (rms, data[name]["mean_delta"]), fontsize=7,
               textcoords="offset points", xytext=(ox, oy), ha='left')

ax.set_xscale('log')
ax.set_xlabel("RMS Scale (hidden state magnitude)", fontsize=11)
ax.set_ylabel("Steering Effectiveness $\\Delta$", fontsize=11)
ax.set_title(f"RMS Scale vs. Steering ($\\rho={r_val:.3f}$, $p={p_val:.4f}$, $n=14$)", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIGS / "fig_pr_scatter.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig_pr_scatter.png")

print("\n" + "="*60)
print("ALL FIGURES REGENERATED WITH 14 MODELS")
print("="*60)
