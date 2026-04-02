import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity

MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen3-0.6B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
    "unsloth_Llama-3.2-1B-Instruct",
    "unsloth_gemma-2-2b-it",
]

MODEL_SHORT = {
    "Qwen_Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen_Qwen3-0.6B": "Qwen3-0.6B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "unsloth_Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "unsloth_gemma-2-2b-it": "Gemma-2-2B",
}

FRAMEWORKS = {
    "bigfive": [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ],
    "mbti": ["extraversion_mbti", "sensing", "thinking", "judging"],
    "jungian": ["ni", "ne", "si", "se", "ti", "te", "fi", "fe"],
}

TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Consc.",
    "extraversion": "Extrav.",
    "agreeableness": "Agree.",
    "neuroticism": "Neuro.",
    "extraversion_mbti": "E/I",
    "sensing": "S/N",
    "thinking": "T/F",
    "judging": "J/P",
    "ni": "Ni",
    "ne": "Ne",
    "si": "Si",
    "se": "Se",
    "ti": "Ti",
    "te": "Te",
    "fi": "Fi",
    "fe": "Fe",
}

FW_COLORS = {
    "bigfive": "#2196F3",
    "mbti": "#FF5722",
    "jungian": "#4CAF50",
}

OUTDIR = "paper/figures"
os.makedirs(OUTDIR, exist_ok=True)


def load_analysis(model, trait):
    path = f"results/persona_vectors/{model}/{trait}/analysis_v2_{trait}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_vec(model, trait, layer):
    path = (
        f"results/persona_vectors/{model}/{trait}/vectors/mean_diff_layer_{layer}.npy"
    )
    if os.path.exists(path):
        return np.load(path)
    path2 = f"results/activations/{model}/{trait}/pos_layer_{layer}.npy"
    neg_path2 = f"results/activations/{model}/{trait}/neg_layer_{layer}.npy"
    if os.path.exists(path2) and os.path.exists(neg_path2):
        pos = np.load(path2)
        neg = np.load(neg_path2)
        return np.mean(pos, axis=0) - np.mean(neg, axis=0)
    return None


def plot_layer_profile(model, trait, ax, color="#2196F3"):
    data = load_analysis(model, trait)
    if data is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{TRAIT_LABELS.get(trait, trait)}")
        return False
    layers = sorted([int(k) for k in data["layers"].keys()])
    n_layers = data["n_layers"]
    norm_layers = [l / n_layers for l in layers]
    ds = [data["layers"][str(l)]["cohens_d"] for l in layers]
    accs = [data["layers"][str(l)]["loso_accuracy"] for l in layers]
    ax2 = ax.twinx()
    ax.plot(norm_layers, ds, color=color, linewidth=1.8, label="Cohen's d")
    ax2.plot(
        norm_layers,
        accs,
        color=color,
        linewidth=1.2,
        linestyle="--",
        alpha=0.6,
        label="LOSO acc",
    )
    bl = data["best_layer_loso"] / n_layers
    ax.axvline(bl, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer", fontsize=8)
    ax.set_ylabel("Cohen's d", fontsize=7, color=color)
    ax2.set_ylabel("LOSO Acc", fontsize=7, color="gray")
    ax2.set_ylim(0, 1.1)
    ax.set_title(
        f"{TRAIT_LABELS.get(trait, trait)} (n={data['n_samples']})", fontsize=9
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    return True


def generate_framework_layer_profiles():
    for fw, traits in FRAMEWORKS.items():
        if fw == "bigfive":
            continue
        for model in MODELS:
            n_traits = len(traits)
            fig, axes = plt.subplots(
                2, (n_traits + 1) // 2, figsize=(4 * ((n_traits + 1) // 2), 6)
            )
            axes = axes.flatten()
            color = FW_COLORS[fw]
            for i, trait in enumerate(traits):
                plot_layer_profile(model, trait, axes[i], color=color)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            mshort = MODEL_SHORT[model]
            fw_display = {"mbti": "MBTI", "jungian": "Jungian Cognitive Functions"}[fw]
            fig.suptitle(
                f"{fw_display} Layer Profiles — {mshort}",
                fontsize=11,
                fontweight="bold",
            )
            plt.tight_layout()
            fname = f"{OUTDIR}/layer_profile_{model}_{fw}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {fname}")


def generate_cross_model_framework_summary():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fw_names = ["bigfive", "mbti", "jungian"]
    fw_display = ["Big Five", "MBTI", "Jungian"]

    for col, (fw, fw_label) in enumerate(zip(fw_names, fw_display)):
        ax = axes[col]
        traits = FRAMEWORKS[fw]
        x = np.arange(len(traits))
        width = 0.15
        color = FW_COLORS[fw]
        colors_model = plt.cm.Set1(np.linspace(0, 0.8, len(MODELS)))

        for mi, model in enumerate(MODELS):
            ds = []
            for trait in traits:
                data = load_analysis(model, trait)
                if data is None:
                    ds.append(0)
                    continue
                bl = str(data["best_layer_loso"])
                d = data["layers"].get(bl, {}).get("cohens_d", 0)
                ds.append(d)
            offset = (mi - 2) * width
            ax.bar(
                x + offset,
                ds,
                width,
                label=MODEL_SHORT[model],
                color=colors_model[mi],
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([TRAIT_LABELS.get(t, t) for t in traits], fontsize=8)
        ax.set_ylabel("Cohen's d (best layer)", fontsize=9)
        ax.set_title(f"{fw_label}", fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Cross-Model Cohen's d by Framework and Trait", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fname = f"{OUTDIR}/cross_model_framework_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def generate_jungian_layer_heatmap():
    model = "Qwen_Qwen3-0.6B"
    traits = FRAMEWORKS["jungian"]
    data_all = {}
    max_layers = 0
    for trait in traits:
        data = load_analysis(model, trait)
        if data:
            data_all[trait] = data
            max_layers = max(max_layers, data["n_layers"])

    norm_bins = np.linspace(0, 1, 20)
    matrix = np.zeros((len(traits), 19))

    for ti, trait in enumerate(traits):
        if trait not in data_all:
            continue
        data = data_all[trait]
        n_layers = data["n_layers"]
        for l_idx, l in enumerate(sorted([int(k) for k in data["layers"].keys()])):
            norm_l = l / n_layers
            bin_idx = min(int(norm_l * 19), 18)
            d = data["layers"][str(l)]["cohens_d"]
            matrix[ti, bin_idx] = max(matrix[ti, bin_idx], d)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=5)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels([TRAIT_LABELS.get(t, t) for t in traits], fontsize=10)
    ax.set_xticks(range(0, 19, 3))
    ax.set_xticklabels([f"{v:.1f}" for v in norm_bins[::3]], fontsize=8)
    ax.set_xlabel("Normalized Layer Position", fontsize=10)
    ax.set_title(
        "Jungian Cognitive Functions: Layer-wise Cohen's d Heatmap (Qwen3-0.6B)",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Cohen's d")
    for ti, trait in enumerate(traits):
        if trait not in data_all:
            continue
        data = data_all[trait]
        bl_norm = data["best_layer_loso"] / data["n_layers"]
        bl_bin = min(int(bl_norm * 19), 18)
        ax.plot(bl_bin, ti, "k*", markersize=10)

    introverted = ["ni", "si", "ti", "fi"]
    extraverted = ["ne", "se", "te", "fe"]
    intro_rows = [traits.index(t) for t in introverted if t in traits]
    extro_rows = [traits.index(t) for t in extraverted if t in traits]
    if intro_rows and extro_rows:
        ax.axhline(max(extro_rows) + 0.5, color="white", linewidth=2, linestyle="--")

    ax.text(
        0.02,
        0.95,
        "Extraverted",
        transform=ax.transAxes,
        fontsize=8,
        color="navy",
        va="top",
        style="italic",
    )
    ax.text(
        0.02,
        0.45,
        "Introverted",
        transform=ax.transAxes,
        fontsize=8,
        color="navy",
        va="top",
        style="italic",
    )
    ax.text(18.5, -0.7, "★ = peak layer", fontsize=7, ha="right")
    plt.tight_layout()
    fname = f"{OUTDIR}/jungian_layer_heatmap.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def generate_mbti_vs_bigfive_extraversion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, (fw, trait, label) in enumerate(
        [
            ("bigfive", "extraversion", "Big Five Extraversion"),
            ("mbti", "extraversion_mbti", "MBTI E/I Dimension"),
        ]
    ):
        ax = axes[col]
        colors = plt.cm.Set1(np.linspace(0, 0.8, len(MODELS)))
        for mi, model in enumerate(MODELS):
            data = load_analysis(model, trait)
            if data is None:
                continue
            layers = sorted([int(k) for k in data["layers"].keys()])
            norm_layers = [l / data["n_layers"] for l in layers]
            ds = [data["layers"][str(l)]["cohens_d"] for l in layers]
            ax.plot(
                norm_layers,
                ds,
                color=colors[mi],
                linewidth=1.8,
                label=MODEL_SHORT[model],
                alpha=0.85,
            )

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xlabel("Normalized Layer", fontsize=10)
        ax.set_ylabel("Cohen's d", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.5, 7)

    axes[0].text(
        0.5,
        0.95,
        "All models: d > 0\n(consistent direction)",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )
    axes[1].text(
        0.5,
        0.95,
        "Qwen2.5: all layers d < 0\n(systematic inversion!)",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )

    plt.suptitle(
        "B5 Extraversion vs MBTI E/I: Same Construct, Different Internal Encoding",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fname = f"{OUTDIR}/mbti_vs_b5_extraversion.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def generate_framework_best_layer_distribution():
    fig, ax = plt.subplots(figsize=(10, 5))
    fw_names = list(FRAMEWORKS.keys())
    fw_display = ["Big Five", "MBTI", "Jungian"]
    positions = {fw: [] for fw in fw_names}

    for fw, traits in FRAMEWORKS.items():
        for model in MODELS:
            for trait in traits:
                data = load_analysis(model, trait)
                if data:
                    positions[fw].append(data["best_layer_loso"] / data["n_layers"])

    for i, (fw, fw_label) in enumerate(zip(fw_names, fw_display)):
        vals = positions[fw]
        color = FW_COLORS[fw]
        parts = ax.violinplot([vals], positions=[i], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.scatter([i] * len(vals), vals, color=color, alpha=0.4, s=15, zorder=3)

    ax.set_xticks(range(len(fw_names)))
    ax.set_xticklabels(fw_display, fontsize=11)
    ax.set_ylabel("Normalized Best Layer Position", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Network midpoint")
    ax.set_title(
        "Distribution of Optimal Layer Position by Framework\n(all models × all traits)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"{OUTDIR}/framework_best_layer_distribution.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def generate_all():
    print("Generating all figures...")
    generate_framework_layer_profiles()
    generate_cross_model_framework_summary()
    generate_jungian_layer_heatmap()
    generate_mbti_vs_bigfive_extraversion()
    generate_framework_best_layer_distribution()
    print("Done.")


if __name__ == "__main__":
    generate_all()
