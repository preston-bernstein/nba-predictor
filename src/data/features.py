from pathlib import Path
import pandas as pd

from .transform import build_features_df  # <- the pure transformer

IN_PATH = Path("data_cache/games.csv")
OUT_PATH = Path("data_cache/features.csv")

def build_features():
    games = pd.read_csv(IN_PATH, parse_dates=["GAME_DATE"])
    feats = build_features_df(games)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(feats):,} rows -> {OUT_PATH}")

if __name__ == "__main__":
    build_features()
