# Makefile for nba-predictor

.PHONY: default pipeline fetch features train test serve clean rebuild all

DATA = data_cache/games.csv
FEATS = data_cache/features.csv
MODEL = artifacts/model.joblib

# Default target: full cycle ending with tests
default: test

# Explicit pipeline (data → features → train, no tests)
pipeline: fetch features train

# Wipe artifacts and data, then rebuild pipeline
rebuild: clean pipeline

# Run the entire workflow: pipeline + tests + service
all: pipeline test serve

fetch: $(DATA)

$(DATA):
	python -m src.data.fetch --seasons 2024 2025

features: $(FEATS)

$(FEATS): $(DATA)
	python -m src.data.features

train: $(MODEL)

$(MODEL): $(FEATS)
	python -m src.model.train

test: $(MODEL)
	pytest -q

serve: $(MODEL)
	uvicorn src.service.app:app --reload

clean:
	rm -rf artifacts/*.joblib artifacts/*.json data_cache/*.csv data_cache/*.parquet
