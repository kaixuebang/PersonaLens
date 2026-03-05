# PersonaLens 🎭

<div align="center">
A Standardized Framework for Mechanistic Localization and Steering of Personality Traits in LLMs.
</div>

## Overview

**PersonaLens** is an end-to-end interpretability framework designed to mechanistically localize, extract, and steer personality representations within Large Language Models (LLMs). Rather than relying on black-box reinforcement learning or fine-tuning, PersonaLens uses contrastive activation analysis to discover the exact linear directions in internal activation space that encode psychological traits (e.g., the Big Five, Freudian defense mechanisms).

This repository contains the complete reproducible codebase for the **PersonaLens** paper.

## Features

- **Standardized Contrastive Prompts:** Comprehensive scenario pairs for 5 Big Five traits + 9 Defense Mechanisms.
- **Multi-Method Extraction:** Extracts persona directions using Mean Difference, PCA, and rigorous L2-regularized Linear Probes with Leave-One-Scenario-Out (LOSO) cross-validation.
- **Causal Localization Pipeline:** Localizes the causal effect of traits using token-level and component-level activation patching (including random-token control experiments).
- **Steering Evaluation Suite:** Provides out-of-the-box scripts to perform $\alpha$-sweeps for personality steering, measuring both behavioral shifts (keyword scoring) and linguistic fluency trade-offs (Context Perplexity).
- **Cross-Model Validation:** Automated scripts to scale experiments across multiple architectures (e.g., Qwen, LLaMA-family).

## Directory Structure

```text
PersonaLens/
├── src/
│   ├── prompts/           # Contrastive scenarios for Big Five & defenses
│   ├── localization/      # Activation collection & token-/component-level patching
│   ├── extraction/        # Vector extraction (Mean Diff, PCA, Linear Probes)
│   ├── steering/          # Activation injection (steering generation)
│   └── evaluation/        # OOD generalization, orthogonality, and final metrics
├── scripts/
│   ├── run_pipeline.py                  # One-click pipeline for a single model
│   └── run_cross_model_experiments.py   # Multi-model scale-up & visual generation 
├── paper/                 # LaTeX sources and PDF of the research paper
├── activations_v2/        # [Generated] Raw contrastive hidden states
├── persona_vectors_v2/    # [Generated] Extracted vectors & LOSO/SNR metrics
├── localization_v2/       # [Generated] Causal patching results
├── cross_model_results/   # [Generated] Cross-model comparison plots
└── steering_results/      # [Generated] Steering evaluations & PPL trade-offs
```

## Setup & Installation

```bash
git clone https://github.com/yourusername/personalens.git
cd personalens

# Install dependencies
pip install torch transformers numpy matplotlib scipy scikit-learn tqdm
```

## Quick Start (Single Model)

You can run the full analytical pipeline on a small prototyping model (e.g., `Qwen/Qwen2.5-0.5B-Instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) locally:

```bash
# Run the complete pipeline for Openness
python scripts/run_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct --trait openness

# Run for all Big Five traits
python scripts/run_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct --trait big5
```

## Scaling Up: Running on Remote Servers (7B+ Models)

Because models like `LLaMA-3-8B` or `Qwen2.5-7B` require significant VRAM (especially for activation caching and patching), you should upload this repository to a remote server with adequate GPU resources (e.g., 1x A100 80GB or 2x RTX 3090/4090).

### 1. Uploading the Code
Since the repository is already clean (thanks to `.gitignore`), you can simply clone your repository on the remote server or `scp`/`rsync` the folder:

```bash
rsync -avz --exclude '.git' --exclude 'activations*' --exclude 'persona_vectors*' ./personalens user@remote_server:/path/to/remote/
```

### 2. Running the Large Scale Experiments
Once on the server, you can use the multi-model script to automatically generate all the artifacts required for the paper, iterating through specified massive models.

```bash
# Example: Running the full suite on a 7B model
python scripts/run_cross_model_experiments.py \
    --models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct" \
    --traits "openness,conscientiousness,extraversion"
```

## Research Methodology

Our framework follows a five-phase methodology (detailed in `paper/main.pdf`):
1. **Contrastive Data Construction** (`src/prompts/`): High vs. low trait personas + neutral scenarios.
2. **Representation Extraction** (`src/extraction/`): Calculating the difference in activation distributions. Evaluated using Cohen's $d$ and rigorous cross-validation.
3. **Causal Localization** (`src/localization/`): Corrupting and restoring activation streams to distinguish genuine causal nodes from merely correlated ones.
4. **Behavioral Steering** (`src/steering/`): Directly shifting model behavior via addition of the persona vector during autoregressive generation.
5. **Evaluation & Visualization** (`src/evaluation/`): Computing trait orthogonality ($\cos$ similarities), OOD generalization bounds, and fluency-behavior inverted-U trade-offs.

## Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{personalens2026,
  title={PersonaLens: A Standardized Framework for Mechanistic Localization and Steering of Personality Traits in Large Language Models},
  author={Anonymous Authors},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
