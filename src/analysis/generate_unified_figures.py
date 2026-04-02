import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

FW_COLORS = {
    "bigfive": "#2196F3",  # Blue
    "mbti": "#FF5722",  # Red-Orange
    "jungian": "#4CAF50",  # Green
}

MODEL_COLORS = {
    "Qwen_Qwen2.5-0.5B-Instruct": "#e41a1c",
    "Qwen_Qwen3-0.6B": "#377eb8",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "#4daf4a",
    "unsloth_Llama-3.2-1B-Instruct": "#984ea3",
    "unsloth_gemma-2-2b-it": "#ff7f00",
}

MODEL_SHORT = {
    "Qwen_Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen_Qwen3-0.6B": "Qwen3-0.6B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "unsloth_Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "unsloth_gemma-2-2b-it": "Gemma-2-2B",
}


MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen3-0.6B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
    "unsloth_Llama-3.2-1B-Instruct",
    "unsloth_gemma-2-2b-it",
]

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

FRAMEWORK_DISPLAY = {
    "bigfive": "Big Five",
    "mbti": "MBTI",
    "jungian": "Jungian Cognitive Functions",
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
    "humor": "Humor",
    "sublimation": "Sublimation",
    "rationalization": "Rationalization",
    "intellectualization": "Intellectualization",
    "displacement": "Displacement",
    "projection": "Projection",
    "denial": "Denial",
    "regression": "Regression",
    "reaction_formation": "Reaction Form.",
}

BIGFIVE_TRAITS = FRAMEWORKS["bigfive"]
MBTI_TRAITS = FRAMEWORKS["mbti"]
JUNGIAN_TRAITS = FRAMEWORKS["jungian"]
ALL_PERSONA_TRAITS = BIGFIVE_TRAITS + MBTI_TRAITS + JUNGIAN_TRAITS

DEFENSE_TRAITS = [
    "humor",
    "sublimation",
    "rationalization",
    "intellectualization",
    "displacement",
    "projection",
    "denial",
    "regression",
    "reaction_formation",
]

DEFENSE_LEVELS = {
    "humor": "mature",
    "sublimation": "mature",
    "rationalization": "neurotic",
    "intellectualization": "neurotic",
    "displacement": "neurotic",
    "projection": "immature",
    "denial": "immature",
    "regression": "immature",
    "reaction_formation": "immature",
}

DEFENSE_LEVEL_ORDER = ["mature", "neurotic", "immature"]
DEFENSE_LEVEL_TRAITS = {
    lvl: [trait for trait in DEFENSE_TRAITS if DEFENSE_LEVELS[trait] == lvl]
    for lvl in DEFENSE_LEVEL_ORDER
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PERSONA_DIR = RESULTS_DIR / "persona_vectors"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
LOCALIZATION_DIR = RESULTS_DIR / "localization"
OUTDIR = REPO_ROOT / "paper" / "figures"


def log(msg: str) -> None:
    print(f"[unified_figures] {msg}")


def warn(msg: str) -> None:
    print(f"[unified_figures][WARN] {msg}")


def save_figure(
    fig: plt.Figure, out_path: Path, rect: Optional[Sequence[float]] = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def save_placeholder(out_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, color="dimgray")
    save_figure(fig, out_path)


def safe_json_load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warn(f"Failed to read JSON: {path} ({exc})")
        return None


def analysis_path(model: str, trait: str) -> Path:
    return PERSONA_DIR / model / trait / f"analysis_v2_{trait}.json"


def vector_path(model: str, trait: str, layer: int) -> Path:
    return PERSONA_DIR / model / trait / "vectors" / f"mean_diff_layer_{layer}.npy"


def load_analysis(model: str, trait: str) -> Optional[dict]:
    return safe_json_load(analysis_path(model, trait))


def list_layers(analysis: dict) -> List[int]:
    layers = analysis.get("layers", {})
    layer_ids: List[int] = []
    for key in layers.keys():
        try:
            layer_ids.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(layer_ids)


def norm_depth(layer: int, n_layers: int) -> float:
    denom = max(n_layers - 1, 1)
    return float(layer) / float(denom)


def get_metric_series(analysis: dict, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    layers = list_layers(analysis)
    if not layers:
        return np.array([]), np.array([])
    n_layers = int(analysis.get("n_layers", max(layers) + 1))
    x = np.array([norm_depth(layer, n_layers) for layer in layers], dtype=float)
    y = np.array(
        [
            float(analysis["layers"].get(str(layer), {}).get(metric, np.nan))
            for layer in layers
        ],
        dtype=float,
    )
    return x, y


def best_layer(analysis: dict) -> Optional[int]:
    if "best_layer_loso" in analysis:
        try:
            return int(analysis["best_layer_loso"])
        except (TypeError, ValueError):
            pass
    if "best_layer_snr" in analysis:
        try:
            return int(analysis["best_layer_snr"])
        except (TypeError, ValueError):
            pass

    layers = list_layers(analysis)
    if not layers:
        return None

    best = max(
        layers,
        key=lambda l: float(analysis["layers"].get(str(l), {}).get("cohens_d", -1e9)),
    )
    return int(best)


def best_layer_cohens_d(analysis: dict) -> float:
    layer = best_layer(analysis)
    if layer is None:
        return float("nan")
    return float(analysis.get("layers", {}).get(str(layer), {}).get("cohens_d", np.nan))


def load_best_vector(model: str, trait: str) -> Optional[np.ndarray]:
    analysis = load_analysis(model, trait)
    if analysis is None:
        return None

    candidates: List[int] = []
    for key in ["best_layer_loso", "best_layer_snr"]:
        if key in analysis:
            try:
                candidates.append(int(analysis[key]))
            except (TypeError, ValueError):
                pass

    layers = list_layers(analysis)
    if layers:
        best_d_layer = max(
            layers,
            key=lambda l: float(
                analysis["layers"].get(str(l), {}).get("cohens_d", -1e9)
            ),
        )
        candidates.append(int(best_d_layer))

    seen = set()
    ordered_candidates = []
    for layer in candidates:
        if layer not in seen:
            ordered_candidates.append(layer)
            seen.add(layer)

    for layer in ordered_candidates:
        path = vector_path(model, trait, layer)
        if path.exists():
            vec = np.load(path).astype(np.float64).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm

    vec_dir = PERSONA_DIR / model / trait / "vectors"
    if vec_dir.exists():
        files = sorted(vec_dir.glob("mean_diff_layer_*.npy"))
        for f in files:
            vec = np.load(f).astype(np.float64).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm

    return None


def plot_single_trait_panel(
    ax: plt.Axes,
    model: str,
    trait: str,
    color: str,
    show_legend: bool = True,
) -> bool:
    analysis = load_analysis(model, trait)
    if analysis is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel("Cohen's d")
        return False

    x, ds = get_metric_series(analysis, "cohens_d")
    _, accs = get_metric_series(analysis, "loso_accuracy")

    if x.size == 0:
        ax.text(0.5, 0.5, "No layers", ha="center", va="center", transform=ax.transAxes)
        return False

    ax2 = ax.twinx()
    line_d = ax.plot(x, ds, color=color, linewidth=1.8, label="Cohen's d")[0]
    line_acc = ax2.plot(
        x,
        accs,
        color=color,
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label="LOSO acc",
    )[0]

    bl = best_layer(analysis)
    if bl is not None:
        n_layers = int(analysis.get("n_layers", int(np.max(x) + 1)))
        ax.axvline(norm_depth(bl, n_layers), color="gray", linestyle=":", alpha=0.6)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)

    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Cohen's d", color=color)
    ax2.set_ylabel("LOSO accuracy", color="gray")
    ax2.set_ylim(0.0, 1.05)

    n_samples = analysis.get("n_samples", "?")
    ax.set_title(f"{TRAIT_LABELS.get(trait, trait)} (n={n_samples})")

    if show_legend:
        ax.legend(
            [line_d, line_acc],
            ["Cohen's d", "LOSO acc"],
            loc="upper left",
            frameon=False,
        )

    return True


def print_data_availability_matrix() -> Dict[str, Dict[str, bool]]:
    log("=== Data availability matrix (analysis_v2 JSON) ===")
    availability: Dict[str, Dict[str, bool]] = {model: {} for model in MODELS}

    for framework, traits in FRAMEWORKS.items():
        print()
        log(f"Framework: {framework} ({len(traits)} traits)")
        header = (
            "Model".ljust(36)
            + " | "
            + " | ".join([TRAIT_LABELS.get(t, t).ljust(8) for t in traits])
        )
        print(header)
        print("-" * len(header))

        for model in MODELS:
            flags = []
            for trait in traits:
                has_data = analysis_path(model, trait).exists()
                availability[model][trait] = has_data
                flags.append(("✓" if has_data else "✗").center(8))
            row = model.ljust(36) + " | " + " | ".join(flags)
            print(row)

    print()
    log("Defense trait availability (for Group 5)")
    header = (
        "Model".ljust(36)
        + " | "
        + " | ".join([TRAIT_LABELS[t].ljust(8) for t in DEFENSE_TRAITS])
    )
    print(header)
    print("-" * len(header))
    for model in MODELS:
        flags = []
        for trait in DEFENSE_TRAITS:
            has_data = analysis_path(model, trait).exists()
            availability[model][trait] = has_data
            flags.append(("✓" if has_data else "✗").center(8))
        print(model.ljust(36) + " | " + " | ".join(flags))

    print()
    return availability


def generate_group1_layer_profiles() -> None:
    log("Group 1: per-model per-framework layer profiles")

    for model in MODELS:
        for framework, traits in FRAMEWORKS.items():
            n_traits = len(traits)
            ncols = 3 if n_traits >= 5 else 2
            nrows = int(math.ceil(n_traits / ncols))

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False
            )
            axes_flat = axes.flatten()

            for idx, trait in enumerate(traits):
                plot_single_trait_panel(
                    axes_flat[idx],
                    model=model,
                    trait=trait,
                    color=FW_COLORS[framework],
                    show_legend=(idx == 0),
                )

            for idx in range(n_traits, len(axes_flat)):
                axes_flat[idx].axis("off")

            fig.suptitle(
                f"{FRAMEWORK_DISPLAY[framework]} Layer Profiles — {MODEL_SHORT.get(model, model)}",
                fontsize=11,
                fontweight="bold",
            )
            out_path = OUTDIR / f"layer_profile_{model}_{framework}.png"
            save_figure(fig, out_path, rect=[0, 0, 1, 0.96])

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_single_trait_panel(
            ax,
            model=model,
            trait="openness",
            color=FW_COLORS["bigfive"],
            show_legend=True,
        )
        fig.suptitle(
            f"Openness Layer Profile — {MODEL_SHORT.get(model, model)}",
            fontsize=11,
            fontweight="bold",
        )
        out_path = OUTDIR / f"layer_profile_{model}_openness.png"
        save_figure(fig, out_path, rect=[0, 0, 1, 0.94])


ORTHO_CACHE: Dict[str, Tuple[List[str], np.ndarray]] = {}


def load_combined_ortho_from_disk(model: str) -> Optional[Tuple[List[str], np.ndarray]]:
    candidates = [
        ANALYSIS_DIR / f"ortho_matrix_all_{model}.npy",
        ANALYSIS_DIR / f"ortho_matrix_persona_{model}.npy",
        ANALYSIS_DIR / f"ortho_matrix_combined_{model}.npy",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            matrix = np.load(path)
        except Exception as exc:
            warn(f"Failed to load orthogonality matrix {path}: {exc}")
            continue

        if (
            matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1]
            and matrix.shape[0] == len(ALL_PERSONA_TRAITS)
        ):
            log(f"Loaded precomputed combined orthogonality matrix: {path}")
            return ALL_PERSONA_TRAITS, matrix.astype(float)

        warn(
            f"Ignoring {path} because shape={matrix.shape} is not combined {len(ALL_PERSONA_TRAITS)}x{len(ALL_PERSONA_TRAITS)}"
        )

    return None


def compute_combined_ortho_from_vectors(
    model: str,
) -> Optional[Tuple[List[str], np.ndarray]]:
    vectors: Dict[str, np.ndarray] = {}
    for trait in ALL_PERSONA_TRAITS:
        vec = load_best_vector(model, trait)
        if vec is None:
            warn(
                f"Missing vector for {model}/{trait}; excluded from combined orthogonality"
            )
            continue
        vectors[trait] = vec

    traits = list(vectors.keys())
    if len(traits) < 2:
        return None

    V = np.vstack([vectors[t] for t in traits])
    matrix = V @ V.T
    matrix = np.clip(matrix, -1.0, 1.0)
    return traits, matrix


def get_combined_orthogonality(model: str) -> Optional[Tuple[List[str], np.ndarray]]:
    if model in ORTHO_CACHE:
        return ORTHO_CACHE[model]

    loaded = load_combined_ortho_from_disk(model)
    if loaded is not None:
        ORTHO_CACHE[model] = loaded
        return loaded

    computed = compute_combined_ortho_from_vectors(model)
    if computed is not None:
        ORTHO_CACHE[model] = computed
    return computed


def plot_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    traits: List[str],
    title: str,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
) -> matplotlib.image.AxesImage:
    masked = np.ma.masked_invalid(matrix)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f0f0f0")

    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")

    labels = [TRAIT_LABELS.get(t, t) for t in traits]
    ax.set_xticks(range(len(traits)))
    ax.set_yticks(range(len(traits)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(
                        j, i, "NA", ha="center", va="center", fontsize=6, color="gray"
                    )
                else:
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=color,
                    )

    return im


def generate_group2_model_orthogonality() -> None:
    log("Group 2: per-model combined orthogonality heatmaps")

    for model in MODELS:
        out_path = OUTDIR / f"ortho_{model}.png"
        result = get_combined_orthogonality(model)
        if result is None:
            save_placeholder(
                out_path,
                title=f"Orthogonality — {MODEL_SHORT.get(model, model)}",
                message="Combined persona vectors are not available.",
            )
            continue

        traits, matrix = result
        size = max(8.5, 0.45 * len(traits) + 2.5)
        fig, ax = plt.subplots(figsize=(size, size))
        im = plot_matrix(
            ax,
            matrix,
            traits,
            title=f"Combined Persona Orthogonality — {MODEL_SHORT.get(model, model)}",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            annotate=True,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")
        save_figure(fig, out_path)


def plot_series_with_std(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
) -> None:
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)
    if std.size == mean.size:
        lo = mean - std
        hi = mean + std
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)


def generate_group3_causal_localization() -> None:
    log("Group 3: per-model causal localization (openness)")

    token_colors = {
        "user_tokens": "#377eb8",
        "system_tokens": "#ff7f00",
        "random_tokens": "#4daf4a",
        "full_layer": "#222222",
    }
    comp_colors = {
        "attn_component": "#984ea3",
        "mlp_component": "#e41a1c",
        "full_layer": "#222222",
    }

    for model in MODELS:
        out_path = OUTDIR / f"causal_loc_{model}_openness.png"
        json_path = LOCALIZATION_DIR / model / "refined_openness.json"
        data = safe_json_load(json_path)

        if data is None:
            save_placeholder(
                out_path,
                title=f"Causal localization — {MODEL_SHORT.get(model, model)}",
                message="Openness causal localization data not available.",
            )
            continue

        lengths = [len(v.get("mean", [])) for v in data.values() if isinstance(v, dict)]
        n_layers = max(lengths) if lengths else 0
        if n_layers <= 0:
            save_placeholder(
                out_path,
                title=f"Causal localization — {MODEL_SHORT.get(model, model)}",
                message="No valid layer-wise causal traces in JSON.",
            )
            continue

        x = np.array([norm_depth(i, n_layers) for i in range(n_layers)], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

        ax = axes[0]
        for key in ["user_tokens", "system_tokens", "random_tokens", "full_layer"]:
            if key not in data:
                continue
            mean = np.array(data[key].get("mean", []), dtype=float)
            std = np.array(data[key].get("std", np.zeros_like(mean)), dtype=float)
            if mean.size != n_layers:
                continue
            plot_series_with_std(
                ax,
                x,
                mean,
                std,
                token_colors.get(key, "black"),
                TRAIT_LABELS.get(key, key),
            )

        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel("KL divergence")
        ax.set_title("Token-localized causal importance")
        ax.legend(frameon=False, loc="upper right")

        ax2 = axes[1]
        for key in ["attn_component", "mlp_component", "full_layer"]:
            if key not in data:
                continue
            mean = np.array(data[key].get("mean", []), dtype=float)
            std = np.array(data[key].get("std", np.zeros_like(mean)), dtype=float)
            if mean.size != n_layers:
                continue
            label = {
                "attn_component": "Attention",
                "mlp_component": "MLP",
                "full_layer": "Full layer",
            }.get(key, key)
            plot_series_with_std(
                ax2, x, mean, std, comp_colors.get(key, "black"), label
            )

        ax2.set_xlabel("Normalized layer depth")
        ax2.set_ylabel("KL divergence")
        ax2.set_title("Component-localized causal importance")
        ax2.legend(frameon=False, loc="upper right")

        fig.suptitle(
            f"Causal Localization (Openness) — {MODEL_SHORT.get(model, model)}",
            fontsize=11,
            fontweight="bold",
        )
        save_figure(fig, out_path, rect=[0, 0, 1, 0.95])


def generate_fig_cross_model_profile() -> None:
    out_path = OUTDIR / "fig_cross_model_profile.png"
    fig, ax = plt.subplots(figsize=(8.2, 4.5))

    plotted = 0
    for model in MODELS:
        analysis = load_analysis(model, "openness")
        if analysis is None:
            warn(f"Missing openness analysis for {model}")
            continue

        x, ds = get_metric_series(analysis, "cohens_d")
        if x.size == 0:
            continue

        color = MODEL_COLORS[model]
        ax.plot(x, ds, color=color, linewidth=1.8, label=MODEL_SHORT.get(model, model))
        bl = best_layer(analysis)
        if bl is not None:
            n_layers = int(analysis.get("n_layers", len(ds)))
            xb = norm_depth(bl, n_layers)
            yb = float(
                analysis.get("layers", {}).get(str(bl), {}).get("cohens_d", np.nan)
            )
            if not np.isnan(yb):
                ax.scatter([xb], [yb], color=color, s=20, zorder=5)
        plotted += 1

    if plotted == 0:
        save_placeholder(
            out_path,
            "Cross-model openness profile",
            "No openness analysis data available.",
        )
        return

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Cross-Model Layer Profile — Openness")
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, out_path)


def generate_fig_cross_model_orthogonality() -> None:
    out_path = OUTDIR / "fig_cross_model_orthogonality.png"
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), squeeze=False)
    axes_flat = axes.flatten()

    ims = []
    for idx, model in enumerate(MODELS):
        ax = axes_flat[idx]
        result = get_combined_orthogonality(model)
        if result is None:
            ax.axis("off")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(MODEL_SHORT.get(model, model))
            continue

        traits, matrix = result
        im = plot_matrix(
            ax,
            matrix,
            traits,
            title=MODEL_SHORT.get(model, model),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            annotate=False,
        )
        ims.append(im)

        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    if ims:
        cbar = fig.colorbar(ims[0], ax=axes_flat.tolist(), fraction=0.015, pad=0.01)
        cbar.set_label("Cosine similarity")

    fig.suptitle(
        "Cross-Model Combined Orthogonality Matrices", fontsize=11, fontweight="bold"
    )
    save_figure(fig, out_path, rect=[0, 0, 1, 0.95])


def generate_cross_model_framework_comparison() -> None:
    out_path = OUTDIR / "cross_model_framework_comparison.png"
    fw_order = ["bigfive", "mbti", "jungian"]
    x = np.arange(len(fw_order))
    width = 0.14

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for mi, model in enumerate(MODELS):
        vals = []
        for fw in fw_order:
            trait_vals = []
            for trait in FRAMEWORKS[fw]:
                analysis = load_analysis(model, trait)
                if analysis is None:
                    continue
                d = best_layer_cohens_d(analysis)
                if not np.isnan(d):
                    trait_vals.append(d)
            vals.append(float(np.mean(trait_vals)) if trait_vals else np.nan)

        offset = (mi - (len(MODELS) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            [0.0 if np.isnan(v) else v for v in vals],
            width=width,
            color=MODEL_COLORS[model],
            alpha=0.85,
            label=MODEL_SHORT.get(model, model),
        )

        for bar, v in zip(bars, vals):
            if np.isnan(v):
                bar.set_hatch("//")
                bar.set_alpha(0.25)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([FRAMEWORK_DISPLAY[fw] for fw in fw_order])
    ax.set_ylabel("Mean Cohen's d at best layer")
    ax.set_title("Cross-Model Framework Comparison")
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, out_path)


def generate_framework_best_layer_distribution() -> None:
    out_path = OUTDIR / "framework_best_layer_distribution.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    fw_order = ["bigfive", "mbti", "jungian"]
    vals_by_fw: Dict[str, List[float]] = {fw: [] for fw in fw_order}

    for fw in fw_order:
        for model in MODELS:
            for trait in FRAMEWORKS[fw]:
                analysis = load_analysis(model, trait)
                if analysis is None:
                    continue
                bl = best_layer(analysis)
                if bl is None:
                    continue
                n_layers = int(analysis.get("n_layers", bl + 1))
                vals_by_fw[fw].append(norm_depth(bl, n_layers))

    for idx, fw in enumerate(fw_order):
        vals = vals_by_fw[fw]
        if not vals:
            continue

        violin = ax.violinplot(
            vals, positions=[idx], widths=0.7, showmeans=True, showmedians=True
        )
        for body in violin["bodies"]:
            body.set_facecolor(FW_COLORS[fw])
            body.set_alpha(0.65)
        violin["cmeans"].set_color("black")
        violin["cmedians"].set_color("black")

        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=18,
            color=FW_COLORS[fw],
            alpha=0.5,
            zorder=3,
        )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label="Network midpoint",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(len(fw_order)))
    ax.set_xticklabels([FRAMEWORK_DISPLAY[fw] for fw in fw_order])
    ax.set_ylabel("Normalized best-layer position")
    ax.set_title("Best-Layer Position Distribution by Framework")
    ax.legend(frameon=False)
    save_figure(fig, out_path)


def generate_jungian_layer_heatmap() -> None:
    out_path = OUTDIR / "jungian_layer_heatmap.png"
    model = "Qwen_Qwen3-0.6B"
    traits = JUNGIAN_TRAITS

    trait_data: Dict[str, dict] = {}
    n_layers = 0
    for trait in traits:
        analysis = load_analysis(model, trait)
        if analysis is None:
            continue
        trait_data[trait] = analysis
        n_layers = max(n_layers, int(analysis.get("n_layers", 0)))

    if n_layers <= 0 or not trait_data:
        save_placeholder(
            out_path, "Jungian layer heatmap", "Qwen3 Jungian data not available."
        )
        return

    matrix = np.full((len(traits), n_layers), np.nan)
    best_points: List[Tuple[int, int]] = []

    for ti, trait in enumerate(traits):
        analysis = trait_data.get(trait)
        if analysis is None:
            continue
        for layer in list_layers(analysis):
            if 0 <= layer < n_layers:
                matrix[ti, layer] = float(
                    analysis["layers"].get(str(layer), {}).get("cohens_d", np.nan)
                )
        bl = best_layer(analysis)
        if bl is not None and 0 <= bl < n_layers:
            best_points.append((ti, bl))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.get_cmap("YlGnBu").copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    vmax = (
        float(np.nanpercentile(matrix, 95)) if np.isfinite(np.nanmax(matrix)) else 1.0
    )
    vmax = max(vmax, 1.0)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)

    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels([TRAIT_LABELS[t] for t in traits])
    xticks = list(range(0, n_layers, max(1, n_layers // 7)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{norm_depth(i, n_layers):.2f}" for i in xticks])
    ax.set_xlabel("Normalized layer depth")
    ax.set_title("Jungian Functions × Layers (Qwen3-0.6B)")

    for ti, li in best_points:
        ax.plot(li, ti, marker="*", color="black", markersize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cohen's d")
    save_figure(fig, out_path)


def generate_mbti_vs_b5_extraversion() -> None:
    out_path = OUTDIR / "mbti_vs_b5_extraversion.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), squeeze=False)
    axes_flat = axes.flatten()

    configs = [
        ("extraversion", "Big Five Extraversion"),
        ("extraversion_mbti", "MBTI E/I"),
    ]

    for ax, (trait, title) in zip(axes_flat, configs):
        plotted = 0
        for model in MODELS:
            analysis = load_analysis(model, trait)
            if analysis is None:
                continue
            x, ds = get_metric_series(analysis, "cohens_d")
            if x.size == 0:
                continue
            ax.plot(
                x,
                ds,
                color=MODEL_COLORS[model],
                linewidth=1.7,
                alpha=0.9,
                label=MODEL_SHORT.get(model, model),
            )
            plotted += 1

        if plotted == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel("Cohen's d")
        ax.set_title(title)
        ax.legend(frameon=False)

    fig.suptitle(
        "B5 vs MBTI Extraversion Layer Profiles", fontsize=11, fontweight="bold"
    )
    save_figure(fig, out_path, rect=[0, 0, 1, 0.95])


def generate_bfi_dose_response() -> None:
    out_path = OUTDIR / "bfi_dose_response.png"

    bfi_v1 = RESULTS_DIR / "bfi_behavioral"
    bfi_v2 = RESULTS_DIR / "bfi_behavioral_v2"

    TRAIT_COLORS = {
        "openness": "#2196F3",
        "conscientiousness": "#4CAF50",
        "extraversion": "#FF9800",
        "agreeableness": "#9C27B0",
        "neuroticism": "#F44336",
    }
    MARKERS = {
        "openness": "o",
        "conscientiousness": "s",
        "extraversion": "^",
        "agreeableness": "D",
        "neuroticism": "v",
    }
    BFI_TRAITS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    per_model_data = {}
    if bfi_v2.exists():
        for model_dir in sorted(bfi_v2.iterdir()):
            if not model_dir.is_dir():
                continue
            trait_data = {}
            for trait in BFI_TRAITS:
                f = model_dir / f"responses_{trait}.json"
                obj = safe_json_load(f)
                if obj and "dose_response_judge" in obj:
                    dr = obj["dose_response_judge"]
                    if dr.get("means") and all(m is not None for m in dr["means"]):
                        trait_data[trait] = dr
            if trait_data:
                per_model_data[model_dir.name] = trait_data

    if not per_model_data and bfi_v1.exists():
        for model_dir in sorted(bfi_v1.iterdir()):
            if not model_dir.is_dir():
                continue
            trait_data = {}
            for trait in BFI_TRAITS:
                f = model_dir / f"bfi_behavioral_{trait}.json"
                obj = safe_json_load(f)
                if obj and "dose_response" in obj:
                    dr = obj["dose_response"]
                    if dr.get("means") and all(m is not None for m in dr["means"]):
                        trait_data[trait] = dr
            if trait_data:
                per_model_data[model_dir.name] = trait_data

    if not per_model_data:
        if out_path.exists():
            log(f"Reusing existing {out_path} (no behavioral data)")
            return
        save_placeholder(
            out_path, "BFI dose-response", "No behavioral JSON data found."
        )
        return

    n_models = len(per_model_data)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, trait_data) in zip(axes, per_model_data.items()):
        for trait in BFI_TRAITS:
            if trait not in trait_data:
                continue
            dr = trait_data[trait]
            alphas = np.array(dr["alphas"])
            means = np.array(dr["means"])
            stds = np.array(dr["stds"])

            ax.plot(
                alphas,
                means,
                marker=MARKERS[trait],
                markersize=4,
                linewidth=1.5,
                color=TRAIT_COLORS[trait],
                label=TRAIT_LABELS.get(trait, trait),
            )
            ax.fill_between(
                alphas,
                means - stds,
                means + stds,
                color=TRAIT_COLORS[trait],
                alpha=0.15,
            )

        ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Steering intensity (α)")
        ax.set_title(MODEL_SHORT.get(model_name, model_name), fontsize=9)
        ax.set_ylim(1.0, 5.0)

    axes[0].set_ylabel("Judge rating (1–5)")
    axes[-1].legend(frameon=False, fontsize=7, ncol=1, loc="upper left")

    scoring = "LLM-as-Judge" if bfi_v2.exists() else "Keyword rubric"
    fig.suptitle(
        f"Behavioral BFI Dose-Response ({scoring}): {n_models} Models × 5 Traits",
        fontsize=10,
        y=1.02,
    )
    save_figure(fig, out_path)


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def generate_alpha_sweep_comparison() -> None:
    out_path = OUTDIR / "alpha_sweep_comparison.png"

    sweep_files = sorted(
        (RESULTS_DIR / "steering_results").glob("**/alpha_sweep_*.json")
    )
    if not sweep_files:
        if out_path.exists():
            log(f"Reusing existing {out_path} (no alpha sweep JSON found)")
            return
        save_placeholder(
            out_path, "Alpha sweep comparison", "No alpha sweep JSON files found."
        )
        return

    def n_points(path: Path) -> int:
        obj = safe_json_load(path)
        return len(obj) if isinstance(obj, dict) else 0

    source = max(sweep_files, key=n_points)
    data_obj = safe_json_load(source)
    if data_obj is None or not isinstance(data_obj, dict) or not data_obj:
        if out_path.exists():
            log(f"Reusing existing {out_path} (alpha sweep JSON invalid)")
            return
        save_placeholder(
            out_path, "Alpha sweep comparison", f"Failed to parse {source.name}."
        )
        return

    openness_keywords = {
        "creative",
        "curious",
        "imaginative",
        "novel",
        "explore",
        "innovation",
        "open",
        "ideas",
        "adventure",
    }

    alphas: List[float] = []
    avg_len_chars: List[float] = []
    keyword_counts: List[float] = []
    keyword_density: List[float] = []
    lexical_diversity: List[float] = []

    for alpha_str, rows in data_obj.items():
        try:
            alpha = float(alpha_str)
        except (TypeError, ValueError):
            continue

        if not isinstance(rows, list) or not rows:
            continue

        responses = [str(r.get("response", "")) for r in rows if isinstance(r, dict)]
        if not responses:
            continue

        words = []
        for resp in responses:
            words.extend(_tokenize_words(resp))

        chars = [len(resp) for resp in responses]
        avg_chars = float(np.mean(chars)) if chars else 0.0
        k_count = float(sum(1 for w in words if w in openness_keywords))
        density = 1000.0 * k_count / max(len(words), 1)
        lex_div = float(len(set(words)) / max(len(words), 1))

        alphas.append(alpha)
        avg_len_chars.append(avg_chars)
        keyword_counts.append(k_count)
        keyword_density.append(density)
        lexical_diversity.append(lex_div)

    if not alphas:
        save_placeholder(
            out_path, "Alpha sweep comparison", "No valid alpha points in sweep JSON."
        )
        return

    order = np.argsort(np.array(alphas))
    alphas_arr = np.array(alphas)[order]
    avg_len_arr = np.array(avg_len_chars)[order]
    key_count_arr = np.array(keyword_counts)[order]
    key_density_arr = np.array(keyword_density)[order]
    lex_arr = np.array(lexical_diversity)[order]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(alphas_arr, avg_len_arr, marker="o", color="#377eb8", linewidth=1.8)
    ax1.set_title("Average response length")
    ax1.set_xlabel("α")
    ax1.set_ylabel("Characters")

    ax2.plot(alphas_arr, key_count_arr, marker="s", color="#4daf4a", linewidth=1.8)
    ax2.set_title("Openness keyword count")
    ax2.set_xlabel("α")
    ax2.set_ylabel("Count")

    ax3.plot(alphas_arr, key_density_arr, marker="^", color="#ff7f00", linewidth=1.8)
    ax3.set_title("Keyword density")
    ax3.set_xlabel("α")
    ax3.set_ylabel("Per 1000 words")

    ax4.plot(alphas_arr, lex_arr, marker="D", color="#984ea3", linewidth=1.8)
    ax4.set_title("Lexical diversity")
    ax4.set_xlabel("α")
    ax4.set_ylabel("Unique/Total words")

    source_model = source.parent.parent.name if source.parent.parent else "unknown"
    source_trait = source.parent.name if source.parent else "unknown"
    fig.suptitle(
        f"Alpha Sweep Comparison — {MODEL_SHORT.get(source_model, source_model)} ({source_trait})",
        fontsize=11,
        fontweight="bold",
    )
    save_figure(fig, out_path, rect=[0, 0, 1, 0.96])


def generate_group4_cross_model_summaries() -> None:
    log("Group 4: cross-model summary figures")
    generate_fig_cross_model_profile()
    generate_fig_cross_model_orthogonality()
    generate_cross_model_framework_comparison()
    generate_framework_best_layer_distribution()
    generate_jungian_layer_heatmap()
    generate_mbti_vs_b5_extraversion()
    generate_bfi_dose_response()
    generate_alpha_sweep_comparison()


def compute_defense_matrix(model: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    vectors: Dict[str, np.ndarray] = {}
    for trait in DEFENSE_TRAITS:
        vec = load_best_vector(model, trait)
        if vec is not None:
            vectors[trait] = vec

    n = len(DEFENSE_TRAITS)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, ti in enumerate(DEFENSE_TRAITS):
        if ti in vectors:
            matrix[i, i] = 1.0
        for j, tj in enumerate(DEFENSE_TRAITS):
            if i >= j:
                continue
            if ti in vectors and tj in vectors:
                val = float(np.dot(vectors[ti], vectors[tj]))
                matrix[i, j] = val
                matrix[j, i] = val
    return matrix, vectors


def generate_defense_mechanism_orthogonality_full() -> None:
    out_path = OUTDIR / "defense_mechanism_orthogonality_full.png"
    model = "Qwen_Qwen2.5-0.5B-Instruct"

    matrix, vectors = compute_defense_matrix(model)
    available = set(vectors.keys())

    if len(available) < 2:
        save_placeholder(
            out_path,
            "Defense mechanism orthogonality",
            "Insufficient defense vectors (need at least 2).",
        )
        return

    labels = []
    markers = {"mature": "★", "neurotic": "◆", "immature": "●"}
    for trait in DEFENSE_TRAITS:
        lvl = DEFENSE_LEVELS[trait]
        labels.append(f"{markers[lvl]} {TRAIT_LABELS.get(trait, trait)}")

    fig, ax = plt.subplots(figsize=(10.2, 8.8))
    im = plot_matrix(
        ax,
        matrix,
        DEFENSE_TRAITS,
        title=(
            f"Defense Mechanism Orthogonality (9×9) — {MODEL_SHORT.get(model, model)}\n"
            f"Available vectors: {len(available)}/9"
        ),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        annotate=True,
    )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.axhline(1.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axhline(4.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axvline(1.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axvline(4.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    save_figure(fig, out_path)


def _mean_abs_cross_level(
    vectors: Dict[str, np.ndarray],
    level_a: str,
    level_b: str,
) -> float:
    traits_a = [t for t in DEFENSE_LEVEL_TRAITS[level_a] if t in vectors]
    traits_b = [t for t in DEFENSE_LEVEL_TRAITS[level_b] if t in vectors]
    vals = [
        abs(float(np.dot(vectors[a], vectors[b]))) for a in traits_a for b in traits_b
    ]
    return float(np.mean(vals)) if vals else float("nan")


def generate_defense_hierarchy_crossmodel() -> None:
    out_path = OUTDIR / "defense_hierarchy_crossmodel.png"

    pair_labels = [
        "Mature-Neurotic",
        "Mature-Immature",
        "Neurotic-Immature",
    ]
    pair_levels = [
        ("mature", "neurotic"),
        ("mature", "immature"),
        ("neurotic", "immature"),
    ]

    pair_values: Dict[str, List[float]] = {}
    counts_by_model: Dict[str, Dict[str, int]] = {}

    any_pair_data = False
    for model in MODELS:
        _, vectors = compute_defense_matrix(model)
        counts_by_model[model] = {
            lvl: sum(1 for t in DEFENSE_LEVEL_TRAITS[lvl] if t in vectors)
            for lvl in DEFENSE_LEVEL_ORDER
        }

        vals = []
        for lvl_a, lvl_b in pair_levels:
            val = _mean_abs_cross_level(vectors, lvl_a, lvl_b)
            if not np.isnan(val):
                any_pair_data = True
            vals.append(val)
        pair_values[model] = vals

    if not any_pair_data:
        save_placeholder(
            out_path,
            "Defense hierarchy cross-model",
            "No cross-level defense vector pairs available in current results.",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax_left, ax_right = axes

    x = np.arange(len(pair_labels))
    width = 0.14
    for mi, model in enumerate(MODELS):
        vals = pair_values[model]
        offset = (mi - (len(MODELS) - 1) / 2.0) * width
        bars = ax_left.bar(
            x + offset,
            [0.0 if np.isnan(v) else v for v in vals],
            width,
            color=MODEL_COLORS[model],
            alpha=0.85,
            label=MODEL_SHORT.get(model, model),
        )
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                bar.set_hatch("//")
                bar.set_alpha(0.25)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(pair_labels)
    ax_left.set_ylabel("Mean |cosine similarity|")
    ax_left.set_title("Cross-level defense similarity")
    ax_left.legend(frameon=False, ncol=2)

    x2 = np.arange(len(MODELS))
    bottom = np.zeros(len(MODELS), dtype=float)
    level_colors = {"mature": "#4daf4a", "neurotic": "#ff7f00", "immature": "#e41a1c"}

    for lvl in DEFENSE_LEVEL_ORDER:
        vals = np.array([counts_by_model[m][lvl] for m in MODELS], dtype=float)
        ax_right.bar(
            x2,
            vals,
            bottom=bottom,
            color=level_colors[lvl],
            alpha=0.85,
            label=lvl.capitalize(),
        )
        bottom += vals

    ax_right.set_xticks(x2)
    ax_right.set_xticklabels([MODEL_SHORT[m] for m in MODELS], rotation=25, ha="right")
    ax_right.set_ylabel("# available defense vectors")
    ax_right.set_title("Defense data availability by maturity level")
    ax_right.legend(frameon=False)

    fig.suptitle(
        "Defense Hierarchy Cross-Model Summary", fontsize=11, fontweight="bold"
    )
    save_figure(fig, out_path, rect=[0, 0, 1, 0.95])


def generate_group5_defense_figures() -> None:
    log("Group 5: defense mechanism figures")
    generate_defense_mechanism_orthogonality_full()
    generate_defense_hierarchy_crossmodel()


def find_latest(pattern: str) -> Optional[Path]:
    files = list(REPO_ROOT.glob(pattern))
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files_sorted[0]


def generate_position_swap_control() -> None:
    out_path = OUTDIR / "position_swap_control.png"
    json_path = find_latest("results/**/position_swap_*.json")
    if json_path is None:
        save_placeholder(
            out_path,
            "Position-swap control",
            "No position_swap_*.json found under results/.",
        )
        return

    summary = safe_json_load(json_path)
    if summary is None:
        save_placeholder(
            out_path, "Position-swap control", f"Failed to parse {json_path.name}."
        )
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    conditions = ["normal", "swapped"]
    titles = {
        "normal": "Normal (System=Persona, User=Scenario)",
        "swapped": "Swapped (System=Scenario, User=Persona)",
    }

    line_max = 0.0
    bar_max = 0.0
    for cond in conditions:
        sys_mean = np.array(
            summary.get(cond, {}).get("system_tokens", {}).get("mean", []), dtype=float
        )
        usr_mean = np.array(
            summary.get(cond, {}).get("user_tokens", {}).get("mean", []), dtype=float
        )
        sys_std = np.array(
            summary.get(cond, {})
            .get("system_tokens", {})
            .get("std", np.zeros_like(sys_mean)),
            dtype=float,
        )
        usr_std = np.array(
            summary.get(cond, {})
            .get("user_tokens", {})
            .get("std", np.zeros_like(usr_mean)),
            dtype=float,
        )

        if sys_mean.size:
            line_max = max(line_max, float(np.max(sys_mean + sys_std)))
        if usr_mean.size:
            line_max = max(line_max, float(np.max(usr_mean + usr_std)))

        bar_max = max(
            bar_max,
            float(summary.get(cond, {}).get("total_system_kl", 0.0)),
            float(summary.get(cond, {}).get("total_user_kl", 0.0)),
        )

    line_max = max(line_max * 1.15, 1e-6)
    bar_max = max(bar_max * 1.2, 1e-6)

    for col, cond in enumerate(conditions):
        ax_line = axes[0, col]
        ax_bar = axes[1, col]

        sys_mean = np.array(
            summary.get(cond, {}).get("system_tokens", {}).get("mean", []), dtype=float
        )
        usr_mean = np.array(
            summary.get(cond, {}).get("user_tokens", {}).get("mean", []), dtype=float
        )
        sys_std = np.array(
            summary.get(cond, {})
            .get("system_tokens", {})
            .get("std", np.zeros_like(sys_mean)),
            dtype=float,
        )
        usr_std = np.array(
            summary.get(cond, {})
            .get("user_tokens", {})
            .get("std", np.zeros_like(usr_mean)),
            dtype=float,
        )

        layers = np.arange(max(sys_mean.size, usr_mean.size), dtype=int)
        x = (
            np.array([norm_depth(i, len(layers)) for i in layers], dtype=float)
            if layers.size
            else np.array([])
        )

        if x.size and sys_mean.size == x.size:
            ax_line.plot(
                x, sys_mean, "o-", color="#ff7f00", linewidth=1.8, label="System tokens"
            )
            ax_line.fill_between(
                x, sys_mean - sys_std, sys_mean + sys_std, color="#ff7f00", alpha=0.15
            )
        if x.size and usr_mean.size == x.size:
            ax_line.plot(
                x, usr_mean, "s-", color="#377eb8", linewidth=1.8, label="User tokens"
            )
            ax_line.fill_between(
                x, usr_mean - usr_std, usr_mean + usr_std, color="#377eb8", alpha=0.15
            )

        ax_line.set_ylim(0, line_max)
        ax_line.set_xlabel("Normalized layer depth")
        ax_line.set_ylabel("KL divergence")
        ax_line.set_title(titles[cond])
        ax_line.legend(frameon=False)

        sys_total = float(summary.get(cond, {}).get("total_system_kl", 0.0))
        usr_total = float(summary.get(cond, {}).get("total_user_kl", 0.0))
        bars = ax_bar.bar(
            ["System", "User"],
            [sys_total, usr_total],
            color=["#ff7f00", "#377eb8"],
            alpha=0.85,
        )
        for bar, val in zip(bars, [sys_total, usr_total]):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                val + bar_max * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax_bar.set_ylim(0, bar_max)
        ax_bar.set_ylabel("Total KL")
        ratio = float(summary.get(cond, {}).get("user_dominance_ratio", np.nan))
        ax_bar.set_title(f"User/System ratio: {ratio:.2f}")

    verdict = summary.get("interpretation", {}).get("position_effect", "UNKNOWN")
    fig.suptitle(
        f"Position-Swap Control — Verdict: {verdict}", fontsize=11, fontweight="bold"
    )
    save_figure(fig, out_path, rect=[0, 0, 1, 0.95])


def generate_null_orthogonality() -> None:
    out_path = OUTDIR / "null_orthogonality.png"
    json_path = find_latest(
        "results/null_orthogonality_results/**/null_orthogonality.json"
    )
    if json_path is None:
        save_placeholder(
            out_path, "Null orthogonality", "No null_orthogonality.json found."
        )
        return

    obj = safe_json_load(json_path)
    if obj is None:
        save_placeholder(
            out_path, "Null orthogonality", f"Failed to parse {json_path.name}."
        )
        return

    null_sim = np.array(
        obj.get("null_attributes", {}).get("similarity_matrix", []), dtype=float
    )
    b5_sim = np.array(obj.get("big_five", {}).get("similarity_matrix", []), dtype=float)
    null_names = obj.get("null_attributes", {}).get("names", [])
    b5_names = obj.get("big_five", {}).get("names", [])
    null_off = obj.get("null_attributes", {}).get("off_diagonal", {}).get("values", [])
    b5_off = obj.get("big_five", {}).get("off_diagonal", {}).get("values", [])

    if null_sim.size == 0 or b5_sim.size == 0:
        save_placeholder(
            out_path,
            "Null orthogonality",
            "JSON present but similarity matrices are empty.",
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(null_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax0.set_xticks(range(len(null_names)))
    ax0.set_yticks(range(len(null_names)))
    ax0.set_xticklabels(
        [n.replace("_", "\n") for n in null_names], rotation=45, ha="right"
    )
    ax0.set_yticklabels([n.replace("_", " ") for n in null_names])
    ax0.set_title("Null attributes")
    for i in range(null_sim.shape[0]):
        for j in range(null_sim.shape[1]):
            color = "white" if abs(null_sim[i, j]) > 0.5 else "black"
            ax0.text(
                j,
                i,
                f"{null_sim[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )

    im1 = ax1.imshow(b5_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(b5_names)))
    ax1.set_yticks(range(len(b5_names)))
    ax1.set_xticklabels(
        [TRAIT_LABELS.get(n, n) for n in b5_names], rotation=45, ha="right"
    )
    ax1.set_yticklabels([TRAIT_LABELS.get(n, n) for n in b5_names])
    ax1.set_title("Big Five")
    for i in range(b5_sim.shape[0]):
        for j in range(b5_sim.shape[1]):
            color = "white" if abs(b5_sim[i, j]) > 0.5 else "black"
            ax1.text(
                j,
                i,
                f"{b5_sim[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )

    means = [
        float(np.mean(null_off)) if null_off else 0.0,
        float(np.mean(b5_off)) if b5_off else 0.0,
    ]
    stds = [
        float(np.std(null_off)) if null_off else 0.0,
        float(np.std(b5_off)) if b5_off else 0.0,
    ]
    bars = ax2.bar(
        ["Null", "Big Five"],
        means,
        yerr=stds,
        capsize=4,
        color=["#ff7f00", "#2196F3"],
        alpha=0.85,
    )
    ax2.set_ylabel("Mean off-diagonal |cos|")
    ax2.set_title("Orthogonality comparison")
    for bar, val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    cbar = fig.colorbar(im1, ax=[ax0, ax1], fraction=0.025, pad=0.02)
    cbar.set_label("Cosine similarity")
    save_figure(fig, out_path)


def generate_shuffle_label_baseline() -> None:
    out_path = OUTDIR / "shuffle_label_baseline.png"
    json_path = find_latest(
        "results/shuffle_label_baseline_results/**/shuffle_label_baseline.json"
    )
    if json_path is None:
        save_placeholder(
            out_path, "Shuffle-label baseline", "No shuffle_label_baseline.json found."
        )
        return

    obj = safe_json_load(json_path)
    if obj is None:
        save_placeholder(
            out_path, "Shuffle-label baseline", f"Failed to parse {json_path.name}."
        )
        return

    genuine_mean = float(obj.get("genuine_big5", {}).get("mean_off_diagonal", np.nan))
    shuffle_dist = np.array(
        obj.get("shuffle_baseline", {}).get("distribution", []), dtype=float
    )

    if np.isnan(genuine_mean) or shuffle_dist.size == 0:
        save_placeholder(
            out_path,
            "Shuffle-label baseline",
            "JSON present but required fields are missing.",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax0, ax1 = axes

    ax0.hist(
        shuffle_dist,
        bins=20,
        color="#ff7f00",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax0.axvline(
        genuine_mean,
        color="#2196F3",
        linestyle="--",
        linewidth=2.0,
        label=f"Genuine = {genuine_mean:.3f}",
    )
    ax0.axvline(
        float(np.mean(shuffle_dist)),
        color="#ff7f00",
        linewidth=1.6,
        label=f"Shuffle mean = {np.mean(shuffle_dist):.3f}",
    )
    ax0.set_xlabel("Mean off-diagonal |cos|")
    ax0.set_ylabel("Count")
    ax0.set_title("Shuffle distribution")
    ax0.legend(frameon=False)

    box = ax1.boxplot(
        [shuffle_dist, [genuine_mean]],
        labels=["Shuffled", "Genuine"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    box["boxes"][0].set_facecolor("#ff7f00")
    box["boxes"][0].set_alpha(0.7)
    box["boxes"][1].set_facecolor("#2196F3")
    box["boxes"][1].set_alpha(0.7)
    ax1.set_ylabel("Mean off-diagonal |cos|")
    ax1.set_title("Genuine vs shuffled")

    p_val = float(obj.get("statistics", {}).get("p_value", np.nan))
    if not np.isnan(p_val):
        ax1.text(
            0.5,
            0.03,
            f"p-value = {p_val:.4f}",
            transform=ax1.transAxes,
            ha="center",
            fontsize=9,
        )

    save_figure(fig, out_path)


def generate_interventional_orthogonality() -> None:
    out_path = OUTDIR / "interventional_orthogonality.png"
    json_path = find_latest("results/**/interventional_orthogonality.json")
    if json_path is None:
        save_placeholder(
            out_path,
            "Interventional orthogonality",
            "No interventional_orthogonality.json found.",
        )
        return

    obj = safe_json_load(json_path)
    if obj is None:
        save_placeholder(
            out_path,
            "Interventional orthogonality",
            f"Failed to parse {json_path.name}.",
        )
        return

    delta_obj = obj.get("delta_matrix", {})
    if not isinstance(delta_obj, dict) or not delta_obj:
        save_placeholder(
            out_path,
            "Interventional orthogonality",
            "JSON present but delta_matrix is empty.",
        )
        return

    source_traits = list(delta_obj.keys())
    target_traits = list(next(iter(delta_obj.values())).keys()) if source_traits else []

    matrix = np.zeros((len(source_traits), len(target_traits)), dtype=float)
    for i, s in enumerate(source_traits):
        for j, t in enumerate(target_traits):
            matrix[i, j] = float(delta_obj.get(s, {}).get(t, 0.0))

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    vmax = max(float(np.max(np.abs(matrix))), 0.1)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(target_traits)))
    ax.set_yticks(range(len(source_traits)))
    ax.set_xticklabels(
        [TRAIT_LABELS.get(t, t) for t in target_traits], rotation=45, ha="right"
    )
    ax.set_yticklabels([TRAIT_LABELS.get(s, s) for s in source_traits])
    ax.set_xlabel("Probe target")
    ax.set_ylabel("Injected source vector")
    ax.set_title("Interventional Orthogonality Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > (0.5 * vmax) else "black"
            ax.text(
                j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Δ probe score")
    save_figure(fig, out_path)


def generate_group6_controls() -> None:
    log("Group 6: control experiment figures")
    generate_position_swap_control()
    generate_null_orthogonality()
    generate_shuffle_label_baseline()
    generate_interventional_orthogonality()


def generate_refined_bigfive_overlays() -> None:
    for trait in BIGFIVE_TRAITS:
        out_path = OUTDIR / f"refined_{trait}.png"
        fig, ax = plt.subplots(figsize=(8.0, 4.5))

        plotted = 0
        for model in MODELS:
            analysis = load_analysis(model, trait)
            if analysis is None:
                continue
            x, ds = get_metric_series(analysis, "cohens_d")
            if x.size == 0:
                continue
            ax.plot(
                x,
                ds,
                color=MODEL_COLORS[model],
                linewidth=1.8,
                alpha=0.9,
                label=MODEL_SHORT.get(model, model),
            )
            plotted += 1

        if plotted == 0:
            save_placeholder(
                out_path,
                f"{TRAIT_LABELS[trait]} cross-model profile",
                "No analysis data available.",
            )
            plt.close(fig)
            continue

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel("Cohen's d")
        ax.set_title(f"Cross-Model Layer Profile — {TRAIT_LABELS[trait]}")
        ax.legend(frameon=False, ncol=2)
        save_figure(fig, out_path)


def choose_reference_model_for_trait(trait: str) -> Optional[str]:
    priority = [
        "Qwen_Qwen2.5-0.5B-Instruct",
        "Qwen_Qwen3-0.6B",
        "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
        "unsloth_Llama-3.2-1B-Instruct",
        "unsloth_gemma-2-2b-it",
    ]
    for model in priority:
        if analysis_path(model, trait).exists():
            return model
    return None


def generate_single_model_layer_analysis_bigfive() -> None:
    for trait in BIGFIVE_TRAITS:
        out_path = OUTDIR / f"layer_analysis_v2_{trait}.png"
        model = choose_reference_model_for_trait(trait)
        if model is None:
            save_placeholder(
                out_path,
                f"Layer analysis v2 — {TRAIT_LABELS[trait]}",
                "No model has this trait analysis.",
            )
            continue

        analysis = load_analysis(model, trait)
        if analysis is None:
            save_placeholder(
                out_path,
                f"Layer analysis v2 — {TRAIT_LABELS[trait]}",
                "Analysis JSON not readable.",
            )
            continue

        x, ds = get_metric_series(analysis, "cohens_d")
        _, acc = get_metric_series(analysis, "loso_accuracy")
        if x.size == 0:
            save_placeholder(
                out_path,
                f"Layer analysis v2 — {TRAIT_LABELS[trait]}",
                "No layer metrics present.",
            )
            continue

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
        ax_d, ax_a = axes

        color = MODEL_COLORS[model]
        ax_d.plot(x, ds, color=color, linewidth=1.9)
        ax_d.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax_d.set_xlabel("Normalized layer depth")
        ax_d.set_ylabel("Cohen's d")
        ax_d.set_title("Effect size")

        ax_a.plot(x, acc, color=color, linewidth=1.9, linestyle="--")
        ax_a.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
        ax_a.set_ylim(0.0, 1.05)
        ax_a.set_xlabel("Normalized layer depth")
        ax_a.set_ylabel("LOSO accuracy")
        ax_a.set_title("Probe performance")

        bl = best_layer(analysis)
        if bl is not None:
            n_layers = int(analysis.get("n_layers", len(x)))
            xb = norm_depth(bl, n_layers)
            ax_d.axvline(xb, color="gray", linestyle=":", alpha=0.6)
            ax_a.axvline(xb, color="gray", linestyle=":", alpha=0.6)

        fig.suptitle(
            f"Layer Analysis v2 — {TRAIT_LABELS[trait]} ({MODEL_SHORT.get(model, model)})",
            fontsize=11,
            fontweight="bold",
        )
        save_figure(fig, out_path, rect=[0, 0, 1, 0.95])


def generate_group7_bigfive_figures() -> None:
    log("Group 7: Big Five per-trait figures")
    generate_refined_bigfive_overlays()
    generate_single_model_layer_analysis_bigfive()


def generate_all() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print_data_availability_matrix()

    generators = [
        ("Group 1", generate_group1_layer_profiles),
        ("Group 2", generate_group2_model_orthogonality),
        ("Group 3", generate_group3_causal_localization),
        ("Group 4", generate_group4_cross_model_summaries),
        ("Group 5", generate_group5_defense_figures),
        ("Group 6", generate_group6_controls),
        ("Group 7", generate_group7_bigfive_figures),
    ]

    for name, fn in generators:
        try:
            log(f"Starting {name}")
            fn()
            log(f"Finished {name}")
        except Exception as exc:
            warn(f"{name} failed: {exc}")

    log("Unified figure generation complete.")


if __name__ == "__main__":
    generate_all()
