# 🏀 NBA Predictor

End-to-end, reproducible pipeline to predict NBA game outcomes:
**fetch → features → train → serve → test**. The code is clean, modular, and fully covered by tests.

---

## Quickstart

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip

# 2) Dev install (editable, with tooling)
python -m pip install -e '.[dev]'

# 3) Run the pipeline
make pipeline    # fetch -> features -> train

# 4) Run tests (will train first if needed)
make test        # quiet
make test-verbose
make test-parallel

# 5) Coverage
make test-cov          # terminal + HTML + XML (fails if < COV_MIN)
make cov-open          # open HTML at coverage_html/index.html
make test-cov-gaps     # list uncovered files/lines/branches
make test-cov-diff     # changed-lines gate vs origin/main
```
> Tip: `make help` lists all commands.

---

## Project Layout

```
nba-predictor/
  src/
    config.py                 # central paths (DATA, FEATS, MODEL)
    data/
      fetch.py                # BR scrape orchestration
      br_client.py            # HTTP client (retriable)
      br_parse.py             # HTML -> tidy CSV rows
      elo.py                  # Elo features
      features.py             # CLI wrapper to build features.csv
      transform.py            # pure feature builders (rolling, rest, joins)
    model/
      datasets.py             # dataset splits, pick_features, to_xy
      metrics.py              # fit & score, safe AUC/ACC
      models.py               # REGISTRY + get_models()
      select.py               # pick_best(), write_metrics(), persist_best_model()
      trainer.py              # multi-model orchestrator
      train.py                # CLI: orchestrates datasets/metrics/models/select
    service/
      app.py                  # FastAPI app factory
      routes.py               # /health, /teams, /predict
      deps.py                 # loaders + matchup_features()
      schemas.py              # Pydantic I/O models
      normalizer.py           # robust team normalization (codes/names/aliases)
      errors.py               # HTTP 422 helper
  tests/                      # 100% coverage, unit + integration
  artifacts/                  # model.joblib, metrics.json (generated)
  data_cache/                 # games.csv, features.csv (generated)
  scripts/
    coverage_gaps.py          # prints remaining coverage holes
  pyproject.toml              # deps, pytest/ruff/mypy config
  Makefile                    # one-liners for everything
```

---

## Pipeline

### 1) Fetch
Scrapes Basketball-Reference and writes `data_cache/games.csv`.

```bash
make fetch
# or: python -m src.data.fetch --seasons "2024 2025"
```

### 2) Features
Builds rolling 10-game form, rest days, and Elo deltas → `data_cache/features.csv`.

```bash
make features
# or: python -m src.data.features
```

### 3) Train
Trains one or more models and persists the best to `artifacts/model.joblib`.

```bash
make train MODELS="logreg rf"
# or: python -m src.model.train --models "logreg rf"
```

Selection uses ROC-AUC, then Accuracy, with NaN-safe comparisons.

---

## Service (FastAPI)

Start dev server:

```bash
make serve
# uvicorn src.service.app:app --reload
```

Endpoints:

- `GET /health` → `{"ok": true}`
- `GET /teams`  → canonical team labels from your historical data
- `GET /predict?home=NYK&away=BOS&date=2025-02-01` → probabilities with feature deltas

**Team normalization** (`service/normalizer.py`):
- Accepts **codes** (`NYK`, `BOS`, `LAL`), **full names** (`New York Knicks`, `Boston Celtics`),
  and **common aliases** (`LA Clippers`, `Golden State`, `Trail Blazers`, …).
- Noise-tolerant to spaces/case/punctuation. Unknown teams → HTTP 422 with a clear message.

**Feature ordering**:
- Prefer `model.feature_columns_` (if persisted at train time).
- Else fall back to `["delta_off","delta_def","delta_rest","delta_elo"]` truncated to `n_features_in_`.

---

## Testing & Coverage

- Tests are exhaustive and isolated; paths are sandboxed by `tests/conftest.py`.
- We gate coverage and provide multiple report styles.

Useful targets:

```bash
make test            # quiet
make test-verbose    # -vv and durations
make test-parallel   # pytest-xdist

make test-cov        # HTML + term-missing + XML
make cov-open        # open HTML report
make test-cov-gaps   # print uncovered files/lines/branches (scripts/coverage_gaps.py)
make test-cov-diff   # changed-lines coverage vs origin/main
```

Notes:
- Right gutter percentages in `pytest-sugar` are **per-file** progress while tests stream.
  The HTML report (`coverage_html/index.html`) and the XML gate are the source of truth.
- We enable **branch coverage** and **per-test** dynamic contexts for deep inspection.

---

## Makefile Knobs

You can override these at the CLI:

```bash
make SEASONS="2024 2025" MODELS="logreg rf" PYTEST_FLAGS="-vv -ra"
make COMPARE_BRANCH="origin/main" test-cov-diff
```

Key vars:
- `SEASONS`: seasons to fetch (space-separated).
- `MODELS`: model list to train (e.g., `"logreg rf"`).
- `PYTEST_FLAGS`: extra pytest flags.
- `COMPARE_BRANCH`: base branch for diff coverage.

Run `make help` for the full menu.

---

## Requirements

Everything is declared in `pyproject.toml`. Install dev tooling with:

```bash
python -m pip install -e '.[dev]'
```

Optional lockfile for CI/Docker:

```bash
make dev-requirements   # produces requirements-dev.txt via pip-tools
```

---

## Notes

- Data source: [Basketball-Reference](https://www.basketball-reference.com/).
- API-based sources exist but are often unreliable for historical consistency; BR is stable.
- This project emphasizes **clarity, determinism, and testability** over model flash.
