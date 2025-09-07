SHELL := /bin/bash
.ONESHELL:

# ---- Config knobs (override at CLI) ----
PY ?= python
SEASONS ?= 2024 2025
MODELS ?= logreg
PYTEST_FLAGS ?= -q
COMPARE_BRANCH ?= origin/main
COV_MIN ?= 85

DATA   := data_cache/games.csv
FEATS  := data_cache/features.csv
MODEL  := artifacts/model.joblib

# Coverage flags (branch coverage + per-test dynamic contexts)
COV_FLAGS = --cov=src --cov-branch --cov-context=test --cov-fail-under=$(COV_MIN)

# ---- Phony ----
.PHONY: default help pipeline fetch features train train-all \
        test test-verbose test-parallel \
        test-cov test-cov-html test-cov-annotate test-cov-json test-cov-diff test-cov-gaps \
        cov-open cov-clean \
        serve clean rebuild all \
        dev dev-requirements lint type fmt

default: test

help:
	@echo "Targets:"
	@echo "  dev                - install package editable w/ dev deps"
	@echo "  fetch|features|train - data/feature/model pipeline"
	@echo "  test               - run pytest (depends on trained model)"
	@echo "  test-verbose       - verbose + durations"
	@echo "  test-parallel      - pytest -n auto (xdist)"
	@echo "  test-cov           - term + HTML + XML coverage"
	@echo "  test-cov-html      - HTML-only (fast rerun)"
	@echo "  test-cov-annotate  - write annotated sources with '!' misses"
	@echo "  test-cov-json      - write coverage.json"
	@echo "  test-cov-diff      - changed-lines coverage vs $(COMPARE_BRANCH)"
	@echo "  test-cov-gaps      - print uncovered files/lines/branches"
	@echo "  cov-open           - open HTML coverage report"
	@echo "  lint / type / fmt  - ruff, mypy, ruff format"

# ---------- Dev setup ----------
dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e '.[dev]'
	@if command -v pre-commit >/dev/null 2>&1; then pre-commit install; fi

dev-requirements:
	$(PY) -m pip install -U pip pip-tools
	pip-compile --extra dev pyproject.toml -o requirements-dev.txt

# ---------- Pipeline ----------
pipeline: fetch features train
rebuild: clean pipeline
all: pipeline test

fetch: $(DATA)
$(DATA):
	$(PY) -m src.data.fetch --seasons $(SEASONS)

features: $(FEATS)
$(FEATS): $(DATA)
	$(PY) -m src.data.features

train: $(MODEL)
$(MODEL): $(FEATS)
	$(PY) -m src.model.train --models $(MODELS)

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

# Human-friendly “what is still uncovered?” summary (uses scripts/coverage_gaps.py)
test-cov-gaps: $(MODEL)
	coverage erase
	pytest -q $(COV_FLAGS) --cov-report=xml:coverage.xml
	$(PY) scripts/coverage_gaps.py

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
