import os
from pathlib import Path

# Base directories (overridable via env)
DATA_DIR = Path(os.getenv("NBA_DATA_DIR", "data_cache"))
ART_DIR = Path(os.getenv("NBA_ART_DIR", "artifacts"))

# Filenames (also overridable)
GAMES_FILE = os.getenv("NBA_GAMES_FILE", "games.csv")
FEATS_FILE = os.getenv("NBA_FEATS_FILE", "features.csv")
MODEL_FILE = os.getenv("NBA_MODEL_FILE", "model.joblib")
METRICS_FILE = os.getenv("NBA_METRICS_FILE", "metrics.json")

# Full paths (convenience)
GAMES = DATA_DIR / GAMES_FILE
FEATS = DATA_DIR / FEATS_FILE
MODEL = ART_DIR / MODEL_FILE

# Canonical training/serving feature order -- single source of truth so
# train.py, trainer.py, and routes.py can't silently drift from each other.
FEATURE_ORDER: tuple[str, ...] = ("delta_off", "delta_def", "delta_rest", "delta_elo")

# Required columns for a raw games dataframe -- single source of truth so
# elo.py, transform.py, and deps.py can't silently drift from each other.
REQUIRED_GAME_COLS: frozenset[str] = frozenset({
    "GAME_DATE",
    "home_team",
    "home_score",
    "away_team",
    "away_score",
})
