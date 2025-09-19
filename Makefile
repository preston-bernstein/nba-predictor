SHELL := /bin/bash
.ONESHELL:

# ---- Config knobs (override at CLI) ----
PY ?= python
SEASONS ?= 2024 2025
MODELS ?= logreg
PYTEST_FLAGS ?= -q
COMPARE_BRANCH ?= origin/main
COV_MIN ?= 85
OFFLINE ?= 0
ifeq ($(CI),true)
  OFFLINE := 1
endif

DATA   := data_cache/games.csv
FEATS  := data_cache/features.csv
MODEL  := artifacts/model.joblib

# Optional fixtures used when OFFLINE=1
FIX_DIR := tests/fixtures
FIX_GAMES := $(FIX_DIR)/games_mini.csv
FIX_FEATS := $(FIX_DIR)/features_mini.csv
FIX_MODEL := $(FIX_DIR)/model_mini.joblib

# Coverage flags (branch coverage + per-test dynamic contexts)
COV_FLAGS = --cov=src --cov-branch --cov-context=test --cov-fail-under=$(COV_MIN)

# ---- Phony ----
.PHONY: default help pipeline fetch features train train-all \
        test test-verbose test-parallel \
        test-cov test-cov-html test-cov-annotate test-cov-json test-cov-diff test-cov-gaps test-cov-gate \
        cov-open cov-clean \
        serve clean rebuild all \
        dev dev-requirements lint type fmt \
        hooks check ci precommit \
        fetch-online fetch-offline features-offline train-offline

default: test

help:
	@echo "Targets:"
	@echo "  dev                  - install package editable w/ dev deps"
	@echo "  fetch|features|train - data/feature/model pipeline"
	@echo "  test                 - run pytest (depends on trained model)"
	@echo "  test-verbose         - verbose + durations"
	@echo "  test-parallel        - pytest -n auto (xdist)"
	@echo "  test-cov             - term + HTML + XML coverage"
	@echo "  test-cov-html        - HTML-only (fast rerun)"
	@echo "  test-cov-annotate    - write annotated sources with '!' misses"
	@echo "  test-cov-json        - write coverage.json"
	@echo "  test-cov-diff        - changed-lines coverage vs $(COMPARE_BRANCH)"
	@echo "  test-cov-gaps        - print uncovered files/lines/branches"
	@echo "  cov-open             - open HTML coverage report"
	@echo "  lint / type / fmt    - ruff, mypy, ruff format"
	@echo "  OFFLINE=1            - use fixtures (no network/scraping); ideal for CI"

# ---------- Dev setup ----------
dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e '.[dev]'
	@if command -v pre-commit >/dev/null 2>&1 && [ -f .pre-commit-config.yaml ]; then pre-commit install; else echo "pre-commit skipped (missing or not installed)"; fi

dev-requirements:
	$(PY) -m pip install -U pip pip-tools
	pip-compile --extra dev pyproject.toml -o requirements-dev.txt

# ---------- Deployment ----------

# Local gate before push
check: fmt lint type test-cov-gate

# CI gate (offline by default; add diff-cover if desired)
ci:
	$(MAKE) OFFLINE=1 test-cov-gate

# Run your pre-commit hooks over the whole repo
precommit:
	pre-commit run --all-files

hooks:
	@if [ -f .pre-commit-config.yaml ]; then pre-commit run --all-files; else echo ".pre-commit-config.yaml not found"; exit 1; fi

# ---------- Pipeline ----------
pipeline: fetch features train
rebuild: clean pipeline
all: pipeline test

# Online/offline split for fetch/features/train
ifeq ($(OFFLINE),1)

fetch: fetch-offline
features: features-offline
train: train-offline

fetch-offline: $(DATA)
$(DATA):
	@echo "OFFLINE=1: seeding games from fixtures -> $(DATA)"
	mkdir -p data_cache
	@if [ -f "$(FIX_GAMES)" ]; then cp "$(FIX_GAMES)" "$(DATA)"; \
	else echo "ERROR: Fixture $(FIX_GAMES) not found. Provide a small sample games CSV."; exit 1; fi

features-offline: $(FEATS)
$(FEATS): $(DATA)
	@echo "OFFLINE=1: seeding features from fixtures -> $(FEATS)"
	mkdir -p data_cache
	@if [ -f "$(FIX_FEATS)" ]; then cp "$(FIX_FEATS)" "$(FEATS)"; \
	else echo "ERROR: Fixture $(FIX_FEATS) not found. Provide a small sample features CSV."; exit 1; fi

train-offline: $(MODEL)
$(MODEL): $(FEATS)
	@echo "OFFLINE=1: seeding model from fixtures -> $(MODEL)"
	mkdir -p artifacts
	@if [ -f "$(FIX_MODEL)" ]; then cp "$(FIX_MODEL)" "$(MODEL)"; \
	else echo "ERROR: Fixture $(FIX_MODEL)" not found. Provide a small pre-trained model.; exit 1; fi

else  # ----------- ONLINE (default local) -----------

fetch: fetch-online
fetch-online: $(DATA)
$(DATA):
	$(PY) -m src.data.fetch --seasons $(SEASONS)

features: $(FEATS)
$(FEATS): $(DATA)
	$(PY) -m src.data.features

train: $(MODEL)
$(MODEL): $(FEATS)
	$(PY) -m src.model.train --models $(MODELS)

endif

train-all:
	$(MAKE) train MODELS="logreg rf"

# ---------- Tests ----------
test: $(MODEL)
	pytest $(PYTEST_FLAGS)

test-verbose: $(MODEL)
	pytest -vv -ra --durations=15

test-parallel: $(MODEL)
	pytest -vv -n auto -ra $(PYTEST_FLAGS)

# ---------- Coverage ----------
# Full run: terminal (missing), HTML, XML; fails if < COV_MIN
test-cov: $(MODEL)
	coverage erase
	pytest -vv -ra $(COV_FLAGS) \
	  --cov-report=term-missing:skip-covered \
	  --cov-report=html:coverage_html \
	  --cov-report=xml:coverage.xml
	@echo "HTML: coverage_html/index.html (make cov-open)"

# Quick HTML re-run (no XML/term noise)
test-cov-html: $(MODEL)
	coverage erase
	pytest -q $(COV_FLAGS) --cov-report=html:coverage_html

test-cov-annotate: $(MODEL)
	coverage erase
	pytest -q $(COV_FLAGS)
	coverage annotate -d coverage_annotate
	@echo "See coverage_annotate/*.py"

test-cov-json: $(MODEL)
	coverage json -o coverage.json

# Changed-lines coverage gate (requires diff-cover)
test-cov-diff: $(MODEL)
	pytest -q $(COV_FLAGS) --cov-report=xml:coverage.xml
	diff-cover coverage.xml --compare-branch=$(COMPARE_BRANCH) --fail-under=$(COV_MIN)

# Human-friendly “what is still uncovered?” summary
test-cov-gaps: $(MODEL)
	coverage erase
	pytest -q $(COV_FLAGS) --cov-report=xml:coverage.xml
	$(PY) scripts/coverage_gaps.py

test-cov-gate: $(MODEL)
	coverage erase
	pytest -vv -ra -n auto \
	  --cov=src --cov-branch --cov-context=test \
	  --cov-report=term-missing:skip-covered \
	  --cov-report=html:coverage_html \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=$${COV_MIN:-95}

cov-open:
	@open coverage_html/index.html 2>/dev/null || $(PY) -c "import webbrowser; webbrowser.open('coverage_html/index.html')"

cov-clean:
	rm -f .coverage coverage.xml coverage.json
	rm -rf coverage_html coverage_annotate

# ---------- Service ----------
serve: $(MODEL)
	uvicorn src.service.app:app --reload

# ---------- QA ----------
lint:
	ruff check .

type:
	mypy src

fmt:
	ruff format .

clean:
	rm -rf artifacts data_cache
	mkdir -p artifacts data_cache
