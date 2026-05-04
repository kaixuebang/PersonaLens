import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
import os

MODELS = [
    ('unsloth_llama-3-8B-Instruct', 14, 'Llama-3-8B'),
    ('Qwen_Qwen2.5-7B-Instruct', 14, 'Qwen2.5-7B'),
    ('mistralai_Mistral-7B-Instruct-v0.1', 8, 'Mistral-7B'),
]
TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
TRAIT_COLORS = {
    'openness': '#2196F3',
    'conscientiousness': '#4CAF50',
    'extraversion': '#F44336',
    'agreeableness': '#FF9800',
    'neuroticism': '#9C27B0',
}
TRAIT_LABELS = {
    'openness': 'O',
    'conscientiousness': 'C',
    'extraversion': 'E',
    'agreeableness': 'A',
    'neuroticism': 'N',
}
ACT_BASE = '/data1/tongjizhou/persona/activations'
OUT_DIR = '/data1/tongjizhou/persona/paper/figures'
os.makedirs(OUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for ax_idx, (model_dir, layer, model_name) in enumerate(MODELS):
    all_acts = []
    all_labels = []
    all_dirs = []

    for trait in TRAITS:
        act_dir = os.path.join(ACT_BASE, model_dir, trait)
        pos_file = os.path.join(act_dir, f'pos_layer_{layer}.npy')
        neg_file = os.path.join(act_dir, f'neg_layer_{layer}.npy')
        if not os.path.exists(pos_file):
            print(f'SKIP {model_name}/{trait}/L{layer}')
            continue
        pos = np.load(pos_file)
        neg = np.load(neg_file)
        combined = np.concatenate([pos, neg], axis=0)
        all_acts.append(combined)
        all_labels.extend([trait] * len(combined))
        all_dirs.extend(['high'] * len(pos) + ['low'] * len(neg))

    X = np.concatenate(all_acts, axis=0)
    labels = np.array(all_labels)
    print(f'{model_name}: {X.shape[0]} samples, dim={X.shape[1]}')

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42, metric='cosine')
    X_2d = umap.fit_transform(X_pca)

    ax = axes[ax_idx]

    for trait in TRAITS:
        mask = labels == trait
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=TRAIT_COLORS[trait],
            label=TRAIT_LABELS[trait],
            alpha=0.35,
            s=8,
            edgecolors='none',
        )

    for trait in TRAITS:
        mask = labels == trait
        cx, cy = X_2d[mask].mean(axis=0)
        ax.scatter(cx, cy, c=TRAIT_COLORS[trait], s=120, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=10)
        ax.annotate(TRAIT_LABELS[trait], (cx, cy),
                    fontsize=11, fontweight='bold',
                    ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points',
                    color=TRAIT_COLORS[trait])

    ax.set_title(model_name, fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if ax_idx == 0:
        handles = []
        for trait in TRAITS:
            h = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=TRAIT_COLORS[trait],
                           markersize=8, label=trait.capitalize())
            handles.append(h)
        ax.legend(handles=handles, loc='lower left', fontsize=8,
                  framealpha=0.9, ncol=1)

axes[0].text(0.02, 0.98, 'Stars = trait centroids',
             transform=axes[0].transAxes, fontsize=8,
             va='top', ha='left', style='italic', color='gray')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'fig_umap_oe_collapse.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
