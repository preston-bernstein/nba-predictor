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
- `GET /v1/ledger/summary` → `{"games", "settled", "graded", "accuracy"}` from the prediction ledger. `graded` counts settled games that have a stored prediction. `accuracy` is null when `graded` is 0.

Team inputs accept codes, full names, and common aliases. Unknown teams return HTTP 422 with a clear message. The service reads artifacts only; regenerate them before deploying.

## Kafka consumer (optional)

Event contract, topics, and how to inspect lag: [nba-infra/kafka/README.md](https://github.com/preston-bernstein/nba-infra/blob/main/kafka/README.md).

The consumer keeps a prediction ledger. It runs only when `KAFKA_BROKERS` is set.

```bash
KAFKA_BROKERS=localhost:29092 python -m src.events.consumer
```

- It reads topic `game.updated` in group `predictor`. It commits each offset by hand, after the work is done. New groups start at the earliest offset.
- The ledger is a SQLite file at `PREDICTION_LEDGER_PATH` (default `$NBA_DATA_DIR/prediction_ledger.sqlite`). The `GET /v1/ledger/summary` endpoint reads the same file.
- A SCHEDULED or IN_PROGRESS event for a new game stores a prediction from the same code as `/v1/predict`. A stored prediction is never replaced.
- A FINAL event stores the scores and outcome and marks the game settled. If there was no prediction, it is stored as null.
- Each `eventId` is applied once. A repeat is skipped.
- A bad message goes to `game.updated.dlq` with `dlq-*` headers, then is committed. Bad means invalid JSON, a schema mismatch, a tied final score, or a team the predictor cannot use.
- Other errors are retried with a growing delay (up to `KAFKA_MAX_RETRIES`, default 5). If they last, the consumer exits without committing, and a restart tries the message again.
- Other settings: `KAFKA_GROUP_ID`, `KAFKA_TOPIC`, `KAFKA_DLQ_TOPIC`.
- The event contract is in `nba-infra/kafka/README.md`. A copy of the schema is in `src/events/game.updated.v1.json`; a test checks it matches `tests/fixtures/game.updated.v1.json`.

Integration tests need a broker: `KAFKA_TEST_BROKERS=localhost:29092 pytest -m integration`. They make and delete their own topics and groups.

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
