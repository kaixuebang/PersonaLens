# Makefile for PersonaLens - Mechanical Interpretability for Personality Traits
# Full automated pipeline from data collection to paper generation

.PHONY: help install test clean pipeline paper all verify shuffle-label-baseline interventional-ortho relative-injection

# Default model and trait
MODEL ?= Qwen/Qwen3-0.6B
TRAIT ?= openness
ACTIVATIONS_DIR ?= results/activations
PERSONA_VECTORS_DIR ?= results/persona_vectors
PAPER_DIR ?= paper

# Python executable
PYTHON ?= python
PIP ?= pip

help:
	@echo "PersonaLens - Automated Research Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install dependencies"
	@echo "  make verify          - Verify environment and dependencies"
	@echo "  make pipeline        - Run full pipeline for default model/trait"
	@echo "  make pipeline-all    - Run pipeline for all models and traits"
	@echo "  make shuffle-label-baseline - Run permutation null-baseline for Big Five orthogonality"
	@echo "  make interventional-ortho   - Test causal disentanglement via cross-probe interventions"
	@echo "  make relative-injection     - Analyze relative injection strength across models"
	@echo "  make tables          - Generate LaTeX tables from results"
	@echo "  make figures         - Collect and organize figures"
	@echo "  make paper           - Compile LaTeX paper"
	@echo "  make full-paper      - Generate tables, figures, and compile paper"
	@echo "  make clean           - Clean generated files"
	@echo "  make clean-all       - Clean everything including activations"
	@echo "  make test            - Run tests"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)"
	@echo "  TRAIT=$(TRAIT)"
	@echo "  ACTIVATIONS_DIR=$(ACTIVATIONS_DIR)"
	@echo "  PERSONA_VECTORS_DIR=$(PERSONA_VECTORS_DIR)"

# Installation
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -e ".[dev]"
	@echo "✓ Development dependencies installed"

# Verification
verify:
	@echo "Verifying environment..."
	@$(PYTHON) -c "import torch; print(f'  PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
	@$(PYTHON) -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
	@$(PYTHON) -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')"
	@which pdflatex > /dev/null 2>&1 && echo "  LaTeX: Available" || echo "  LaTeX: NOT FOUND (required for paper generation)"
	@echo "✓ Environment verification complete"

# Pipeline stages
pipeline:
	@echo "Running full pipeline for $(MODEL) - $(TRAIT)..."
	$(PYTHON) scripts/run_pipeline.py --model $(MODEL) --trait $(TRAIT)
	@echo "✓ Pipeline complete"

pipeline-all:
	@echo "Running pipeline for all traits on $(MODEL)..."
	$(PYTHON) scripts/run_pipeline.py --model $(MODEL) --trait all
	@echo "✓ Pipeline complete"

cross-model:
	@echo "Running cross-model experiments..."
	$(PYTHON) scripts/run_cross_model_experiments.py --models $(MODEL) --trait all
	@echo "✓ Cross-model experiments complete"

# New robustness/rigor experiments
shuffle-label-baseline:
	@echo "Running shuffle-label permutation baseline for $(MODEL)..."
	PYTHONPATH=. $(PYTHON) src/evaluation/eval_shuffle_label_baseline.py \
		--model $(MODEL) --n_permutations 200
	@echo "✓ Shuffle-label baseline complete: results/shuffle_label_baseline_results/"

interventional-ortho:
	@echo "Running interventional orthogonality test for $(MODEL)..."
	PYTHONPATH=. $(PYTHON) src/evaluation/eval_interventional_orthogonality.py \
		--model $(MODEL) --alpha 3.0
	@echo "✓ Interventional orthogonality complete: results/interventional_ortho_results/"

relative-injection:
	@echo "Analyzing relative injection strength across all models..."
	PYTHONPATH=. $(PYTHON) src/evaluation/eval_relative_injection_strength.py
	@echo "✓ Relative injection strength analysis complete: results/relative_injection_results/"

# Data generation
tables:
	@echo "Generating LaTeX tables from experimental results..."
	$(PYTHON) scripts/generate_latex_tables.py --persona_vectors_dir $(PERSONA_VECTORS_DIR) --output_dir $(PAPER_DIR)/tables
	@echo "✓ Tables generated"

figures:
	@echo "Collecting figures..."
	$(PYTHON) scripts/collect_figures.py
	@echo "✓ Figures collected"

update-appendix:
	@echo "Updating LaTeX appendix..."
	$(PYTHON) scripts/update_appendix.py
	@echo "✓ Appendix updated"

# Paper generation
paper:
	@echo "Compiling LaTeX paper..."
	cd $(PAPER_DIR) && \
		pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
		pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@echo "✓ Paper compiled: $(PAPER_DIR)/main.pdf"

full-paper: tables figures update-appendix paper
	@echo "✓ Full paper generation complete"

# Testing
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "✓ Tests complete"

# Linting and formatting
format:
	@echo "Formatting code..."
	black src/ scripts/ --line-length 100
	@echo "✓ Code formatted"

lint:
	@echo "Linting code..."
	flake8 src/ scripts/ --max-line-length 100 --extend-ignore E203,W503
	@echo "✓ Linting complete"

# Cleanup
clean:
	@echo "Cleaning generated files..."
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg
	rm -f $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out $(PAPER_DIR)/*.toc
	rm -f $(PAPER_DIR)/*.synctex.gz $(PAPER_DIR)/*.fdb_latexmk
	rm -f $(PAPER_DIR)/*.fls $(PAPER_DIR)/*.lot $(PAPER_DIR)/*.lof
	rm -rf $(PAPER_DIR)/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Clean complete"

clean-all: clean
	@echo "Cleaning all generated data..."
	rm -rf $(ACTIVATIONS_DIR)/*
	rm -rf $(PERSONA_VECTORS_DIR)/*
	rm -rf results/localization/*
	rm -rf results/steering_results/*
	rm -rf results/cross_model_results/*
	rm -f $(PAPER_DIR)/main.pdf
	@echo "✓ All data cleaned"

# Full automation
all: verify pipeline tables figures paper
	@echo ""
	@echo "=========================================="
	@echo "✓ FULL PIPELINE COMPLETE"
	@echo "=========================================="
	@echo "Results:"
	@echo "  - Activations: $(ACTIVATIONS_DIR)/"
	@echo "  - Persona Vectors: $(PERSONA_VECTORS_DIR)/"	@echo "  - Paper: $(PAPER_DIR)/main.pdf"
	@echo "=========================================="
