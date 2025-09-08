# Makefile for nba-predictor
SHELL := /bin/bash
.ONESHELL:

# ---- Config knobs (override at CLI) ----
PY ?= python
SEASONS ?= 2024 2025
MODELS ?= logreg            # e.g., make train MODELS="logreg rf"
PYTEST_FLAGS ?= -q

DATA   := data_cache/games.csv
FEATS  := data_cache/features.csv
MODEL  := artifacts/model.joblib

# ---- Phony targets ----
.PHONY: default help pipeline fetch features train train-all test serve clean rebuild all dev dev-requirements lint type fmt

# Default: run tests (will build model if needed)
default: test

help:
	@echo "Targets:"
	@echo "  dev               - install package in editable mode with dev deps"
	@echo "  fetch             - fetch raw games (SEASONS=\"$(SEASONS)\")"
	@echo "  features          - build features from games"
	@echo "  train             - train model(s) (MODELS=\"$(MODELS)\") -> best to artifacts/model.joblib"
	@echo "  train-all         - convenience: train logreg and rf"
	@echo "  test              - run pytest (depends on trained model)"
	@echo "  serve             - run FastAPI dev server (uvicorn)"
	@echo "  pipeline          - fetch -> features -> train"
	@echo "  rebuild           - clean -> pipeline"
	@echo "  clean             - remove generated data/artifacts"
	@echo "  lint / type / fmt - ruff, mypy, ruff format"
	@echo "  dev-requirements  - regenerate requirements-dev.txt from pyproject (pip-tools)"

# One-time local setup
dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e '.[dev]'
	@if command -v pre-commit >/dev/null 2>&1; then pre-commit install; fi

# Optional lockfile for CI/Docker (requires pip-tools)
dev-requirements:
	$(PY) -m pip install -U pip pip-tools
	pip-compile --extra dev pyproject.toml -o requirements-dev.txt

# Explicit pipeline (data → features → train)
pipeline: fetch features train

# Wipe artifacts and data, then rebuild pipeline
rebuild: clean pipeline

# Full non-interactive run: pipeline + tests (omit serve so it doesn't block)
all: pipeline test

# ---- Steps & dependencies ----

fetch: $(DATA)
$(DATA):
	$(PY) -m src.data.fetch --seasons $(SEASONS)

features: $(FEATS)
$(FEATS): $(DATA)
	$(PY) -m src.data.features

train: $(MODEL)
$(MODEL): $(FEATS)
	# Train one or more models; best is copied to artifacts/model.joblib
	$(PY) -m src.model.train --models $(MODELS)

# Convenience to compare LR vs RF
train-all:
	$(MAKE) train MODELS="logreg rf"

test: $(MODEL)
	pytest $(PYTEST_FLAGS)

test-verbose: $(MODEL)
	pytest -vv -ra --durations=15

serve: $(MODEL)
	uvicorn src.service.app:app --reload

lint:
	ruff check .

type:
	mypy src

fmt:
	ruff format .

clean:
	rm -rf artifacts data_cache
	mkdir -p artifacts data_cache
