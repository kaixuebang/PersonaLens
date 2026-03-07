# PersonaLens 🎭

<div align="center">

**A Standardized Framework for Mechanistic Localization and Steering of Personality Traits in LLMs**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview

**PersonaLens** is an end-to-end interpretability framework designed to mechanistically localize, extract, and steer personality representations within Large Language Models (LLMs). Rather than relying on black-box reinforcement learning or fine-tuning, PersonaLens uses contrastive activation analysis to discover the exact linear directions in internal activation space that encode psychological traits (e.g., the Big Five, Freudian defense mechanisms).

This repository contains the **complete reproducible codebase** for the PersonaLens paper, with all fixes for the issues identified in the academic audit.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 16GB+ VRAM for 7B models)
- 50GB+ disk space for models and activations
- (Optional) LaTeX installation for paper generation

### Installation

```bash
git clone https://github.com/yourusername/personalens.git
cd personalens

# Install dependencies (recommended: use pinned versions)
pip install -r requirements.txt

# Or install as editable package
pip install -e .

# Verify installation
make verify
```

### One-Command Pipeline

```bash
# Run complete pipeline for a single trait
make pipeline MODEL=Qwen/Qwen2.5-0.5B-Instruct TRAIT=openness

# Run for all Big Five traits
make pipeline MODEL=Qwen/Qwen2.5-0.5B-Instruct TRAIT=big5

# Full automation: pipeline + tables + paper
make all MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

---

## 📁 Directory Structure

```
PersonaLens/
├── src/                      # Source code
│   ├── prompts/              # Contrastive scenarios for Big Five & defenses
│   ├── localization/         # Activation collection & patching
│   ├── extraction/           # Vector extraction with statistical rigor
│   ├── steering/             # Activation injection (steering)
│   └── evaluation/           # OOD generalization & cross-model validation
├── scripts/                  # Automation scripts
│   ├── run_pipeline.py       # One-click pipeline runner
│   ├── run_cross_model_experiments.py
│   ├── generate_latex_tables.py  # Auto-generate tables from results
│   └── cleanup_versions.py   # Clean up old versions
├── paper/                    # LaTeX sources and generated tables
├── tests/                    # Unit tests
├── requirements.txt          # Pinned dependencies
├── pyproject.toml           # Modern Python packaging
├── Makefile                 # Full automation
└── [Generated outputs]       # Created during pipeline execution
    ├── activations/          # Raw contrastive hidden states
    ├── persona_vectors/      # Extracted vectors & LOSO metrics
    ├── localization/         # Causal patching results
    ├── steering_results/     # Steering evaluations
    ├── eval_results/         # Evaluation outputs
    └── cross_model_results/  # Cross-model comparisons
```

**Note**: The `_v2` suffix has been removed. All outputs now use clean, consistent naming. Run `python scripts/cleanup_versions.py` if you have old `_v2` directories from previous runs.

---

## 🔬 Detailed Usage

### Step 1: Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Step 2: Run Analysis Pipeline

```bash
# Option A: Use Makefile (recommended)
make pipeline MODEL=Qwen/Qwen3-0.6B TRAIT=openness

# Option B: Direct Python execution
python scripts/run_pipeline.py \
    --model Qwen/Qwen3-0.6B \
    --trait openness \
    --device cuda
```

The pipeline includes:
1. **Pre-flight checks** - Verify dependencies and environment
2. **Activation collection** - Extract hidden states from contrastive prompts
3. **Persona vector extraction** - Compute directions with LOSO CV and Cohen's d
4. **Causal localization** - Activation patching to identify causal circuits
5. **Steering demonstration** - Generate steered outputs
6. **Cross-model validation** - Compare across architectures
7. **Post-flight verification** - Confirm all outputs generated

### Step 3: Generate Tables and Figures

```bash
# Generate LaTeX tables from experimental results
make tables

# Or manually:
python scripts/generate_latex_tables.py \
    --persona_vectors_dir persona_vectors \
    --output_dir paper/tables
```

This replaces hardcoded tables with **auto-generated** content from actual experimental results.

### Step 4: Compile Paper

```bash
# Full paper generation (tables + figures + compile)
make full-paper

# Or step-by-step:
make tables      # Generate tables
make figures     # Collect figures
make paper       # Compile LaTeX
```

---

## 📊 Statistical Improvements

Based on the academic audit, we've implemented the following fixes:

### 1. Bootstrap Confidence Intervals for Cohen's d

```python
# Now returns: d, ci_lower, ci_upper, p_value
d, ci_lower, ci_upper, p_value = compute_cohens_d(
    pos_acts, neg_acts, 
    compute_ci=True, 
    n_bootstrap=1000,
    ci_level=0.95
)
```

### 2. Permutation Tests for Statistical Significance

- **p-values** computed via permutation testing
- **95% confidence intervals** for all effect sizes
- Results stored in `analysis_v2_{trait}.json`

### 3. Automated Table Generation

- Tables are now **generated from JSON results**
- No more hardcoded values in LaTeX
- Automatic updates when experiments are re-run

---

## 🔧 Advanced Usage

### Multi-Model Experiments

```bash
# Run experiments on multiple models
python scripts/run_cross_model_experiments.py \
    --models "Qwen/Qwen3-0.6B,TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --traits "openness,conscientiousness,extraversion" \
    --output_dir cross_model_results
```

### Custom Steering

```bash
python src/steering/steer_personality.py \
    --model Qwen/Qwen3-0.6B \
    --trait openness \
    --alpha 5.0 \
    --sweep
```

### Skip Slow Steps

```bash
# Skip activation collection (use existing)
python scripts/run_pipeline.py \
    --model Qwen/Qwen3-0.6B \
    --trait openness \
    --skip_collect

# Skip causal localization (fast iteration)
python scripts/run_pipeline.py \
    --model Qwen/Qwen3-0.6B \
    --trait openness \
    --skip_localize
```

---

## 🧪 Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
# 1. Set up environment
make verify

# 2. Run experiments for all models
for model in \
    "Qwen/Qwen3-0.6B" \
    "Qwen/Qwen2.5-0.5B-Instruct" \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"; do
    make pipeline MODEL=$model TRAIT=all
done

# 3. Generate all tables and figures
make tables
make figures

# 4. Compile paper
make paper
```

### Verification Checklist

- [ ] `make verify` passes all checks
- [ ] `activations/{model}/` contains .npy files
- [ ] `persona_vectors/{model}/` contains JSON files with Cohen's d CI
- [ ] `paper/tables/` contains .tex files
- [ ] `paper/main.pdf` compiles without errors

---

## 🐛 Troubleshooting

### Common Issues

**Issue: `System role not supported` error**
- ✅ **Fixed**: Updated `apply_chat_template_safe()` with robust fallback

**Issue: Missing dependencies**
```bash
pip install -r requirements.txt
```

**Issue: CUDA out of memory**
```bash
# Use smaller model or reduce batch size
python scripts/run_pipeline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Issue: LaTeX compilation fails**
```bash
# Install LaTeX
# Ubuntu/Debian:
sudo apt-get install texlive-full

# macOS:
brew install --cask mactex

# Verify:
which pdflatex
```

### Getting Help

```bash
# Show all available make targets
make help

# Run verification
make verify

# Clean and restart
make clean-all
```

---

## 📈 Performance Benchmarks

| Model | VRAM Required | Time (per trait) |
|-------|--------------|------------------|
| Qwen2.5-0.5B | 4GB | ~2 min |
| TinyLlama-1.1B | 6GB | ~3 min |
| Qwen3-0.6B | 5GB | ~3 min |
| LLaMA-3.2-1B | 6GB | ~4 min |
| Gemma-2-2B | 10GB | ~8 min |
| Qwen2.5-7B | 24GB | ~20 min |

---

## 🏗️ Architecture

Our framework follows a five-phase methodology:

1. **Contrastive Data Construction** (`src/prompts/`)
   - High vs. low trait personas
   - 20 scenarios per trait
   - Randomized template selection

2. **Representation Extraction** (`src/extraction/`)
   - Mean Difference, PCA, Linear Probes
   - LOSO cross-validation
   - Cohen's d with 95% CI
   - Permutation p-values

3. **Causal Localization** (`src/localization/`)
   - Token-level activation patching
   - Component-level (MLP/Attention) patching
   - Random-token control experiments

4. **Behavioral Steering** (`src/steering/`)
   - α-sweeps for personality control
   - Keyword-based evaluation
   - Perplexity (fluency) monitoring

5. **Evaluation** (`src/evaluation/`)
   - Cross-model orthogonality
   - OOD generalization
   - Statistical significance testing

---

## 📚 Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{personalens2026,
  title={PersonaLens: A Standardized Framework for Mechanistic Localization 
         and Steering of Personality Traits in Large Language Models},
  author={Anonymous Authors},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This codebase was developed as part of research into mechanistic interpretability for psychological traits in LLMs. The statistical improvements and reproducibility fixes were implemented following the academic audit process.

---

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: research@example.com

---

## ✅ Reproducibility Checklist

- [x] `requirements.txt` with pinned versions
- [x] `pyproject.toml` for modern packaging
- [x] Automated table generation from JSON
- [x] Bootstrap CIs for Cohen's d
- [x] Permutation p-values
- [x] Pre-flight dependency checks
- [x] Post-flight output verification
- [x] Makefile for full automation
- [x] System role template fix
- [x] Comprehensive README

**Status**: ✅ All audit issues addressed
