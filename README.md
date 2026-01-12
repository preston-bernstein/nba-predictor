# NBA predictor

[![CI](https://github.com/preston-bernstein/nba-predictor/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/preston-bernstein/nba-predictor/actions/workflows/ci.yaml) [![Coverage](https://codecov.io/gh/preston-bernstein/nba-predictor/branch/main/graph/badge.svg)](https://codecov.io/gh/preston-bernstein/nba-predictor) [![Python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Python 3.12 pipeline that scrapes Basketball-Reference, builds features, trains a classifier, and serves win probabilities through FastAPI. Cached CSVs in `data_cache/` and artifacts in `artifacts/` drive all predictions; refresh them with the pipeline before serving.

## Setup

Create a virtual environment and install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

## Pipeline

Run everything end to end:

```bash
make pipeline
```

Or run each step:

- Fetch: `make fetch` (or `python -m src.data.fetch --seasons "2024 2025"`) → `data_cache/games.csv`.
- Features: `make features` → rolling form, rest days, and Elo deltas in `data_cache/features.csv`.
- Train: `make train MODELS="logreg rf"` → best model at `artifacts/model.joblib` with metrics in `artifacts/metrics.json`.

Use `OFFLINE=1` to seed from fixtures. Add `PRESERVE=1` to keep existing caches. Control seasons and model lists with `SEASONS` and `MODELS`.

## API

Start the service:

```bash
make serve
# uvicorn src.service.app:app --reload --port 5000
```

Endpoints:

- `GET /v1/health` → `{"ok": true}`
- `GET /v1/teams` → canonical team codes from cached games
- `GET /v1/predict?home=NYK&away=BOS&date=2025-01-01` → win probability and feature deltas

Team inputs accept codes, full names, and common aliases. Unknown teams return HTTP 422 with a clear message. The service reads artifacts only; regenerate them before deploying.

## Tests and QA

Common checks:

```bash
make lint
make type
make test
make test-cov
make check          # fmt + lint + type + coverage gate
```

Coverage runs produce HTML and XML reports. Use `make cov-open`, `make test-cov-gaps`, or `make test-cov-diff` for deeper inspection.

## Layout

```
src/
  config.py           # path config for data and artifacts
  data/               # scraping and feature engineering
  model/              # training and model selection
  service/            # FastAPI app, routes, schemas, normalizer
  utils/              # logging helpers
tests/                # unit and integration tests (fixtures included)
artifacts/            # generated model + metrics
data_cache/           # generated games and features CSVs
Makefile              # pipeline, QA, and serve targets
pyproject.toml        # deps and tooling config
```

Data comes from Basketball-Reference. Treat `data_cache/` and `artifacts/` as generated outputs; rebuild them after code or schema changes.
