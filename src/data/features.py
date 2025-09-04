from pathlib import Path
import pandas as pd

IN_PATH = Path("data_cache/games.csv")
OUT_PATH = Path("data_cache/features.csv")
ROLL = 10  # rolling window (games)

def build_features():
    g = pd.read_csv(IN_PATH, parse_dates=["GAME_DATE"])

    # team-game rows (home, away)
    home = g[["GAME_DATE","home_team","home_score","away_score"]].copy()
    home.columns = ["GAME_DATE","team","pts_for","pts_against"]
    away = g[["GAME_DATE","away_team","away_score","home_score"]].copy()
    away.columns = ["GAME_DATE","team","pts_for","pts_against"]

    tg = pd.concat([home, away], ignore_index=True).sort_values(["team","GAME_DATE"])

    # rolling means from PRIOR games (shift avoids leakage)
    tg["off_r10"] = tg.groupby("team")["pts_for"].transform(lambda s: s.shift().rolling(ROLL, min_periods=3).mean())
    tg["def_r10"] = tg.groupby("team")["pts_against"].transform(lambda s: s.shift().rolling(ROLL, min_periods=3).mean())

    # join back to matchups
    g = g.sort_values("GAME_DATE").copy()
    h = tg.rename(columns={"team":"home_team","off_r10":"home_off_r10","def_r10":"home_def_r10"})[
        ["GAME_DATE","home_team","home_off_r10","home_def_r10"]
    ]
    a = tg.rename(columns={"team":"away_team","off_r10":"away_off_r10","def_r10":"away_def_r10"})[
        ["GAME_DATE","away_team","away_off_r10","away_def_r10"]
    ]
    g = g.merge(h, on=["GAME_DATE","home_team"], how="left").merge(a, on=["GAME_DATE","away_team"], how="left")

    # pregame deltas
    g["delta_off"] = g["home_off_r10"] - g["away_off_r10"]
    g["delta_def"] = g["home_def_r10"] - g["away_def_r10"]

    # enough history only
    g = g.dropna(subset=["delta_off","delta_def"]).reset_index(drop=True)

    feats = g[["GAME_DATE","home_team","away_team","delta_off","delta_def","home_win"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(feats):,} rows -> {OUT_PATH}")

if __name__ == "__main__":
    build_features()
