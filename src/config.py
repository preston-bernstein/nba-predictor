# src/config.py
from pathlib import Path
import os

# Base directories (overridable via env)
DATA_DIR = Path(os.getenv("NBA_DATA_DIR", "data_cache"))
ART_DIR  = Path(os.getenv("NBA_ART_DIR",  "artifacts"))

# Filenames (also overridable)
GAMES_FILE   = os.getenv("NBA_GAMES_FILE",   "games.csv")
FEATS_FILE   = os.getenv("NBA_FEATS_FILE",   "features.csv")
MODEL_FILE   = os.getenv("NBA_MODEL_FILE",   "model.joblib")
METRICS_FILE = os.getenv("NBA_METRICS_FILE", "metrics.json")

# Full paths (convenience)
GAMES   = DATA_DIR / GAMES_FILE
FEATS   = DATA_DIR / FEATS_FILE
MODEL   = ART_DIR  / MODEL_FILE
METRICS = ART_DIR  / METRICS_FILE
