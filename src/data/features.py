# src/data/features.py
from __future__ import annotations
from pathlib import Path
from typing import Final
import pandas as pd
from .elo import add_elo

IN_PATH: Final[Path] = Path("data_cache/games.csv")
OUT_PATH: Final[Path] = Path("data_cache/features.csv")

ROLL: Final[int] = 10          # rolling window (games)
MIN_PERIODS: Final[int] = 3    # min history to compute a rolling stat

def _require_cols(df: pd.DataFrame, cols: set[str], where: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{where}: missing columns: {sorted(missing)}")

def build_features() -> None:
    """
    Build matchup features from raw game rows.

    Input (CSV): data_cache/games.csv with columns
        GAME_DATE, home_team, home_score, away_team, away_score, home_win

    Output (CSV): data_cache/features.csv with columns
        GAME_DATE, home_team, away_team,
        delta_off, delta_def, delta_rest, delta_elo,
        home_win
    """
    g = pd.read_csv(IN_PATH, parse_dates=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)

    _require_cols(
        g,
        {"GAME_DATE", "home_team", "home_score", "away_team", "away_score", "home_win"},
        "features.build",
    )

    # Coerce scores to numeric (fail early if bad)
    g["home_score"] = pd.to_numeric(g["home_score"], errors="raise")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="raise")

    # team-game rows (home & away perspectives)
    home = g[["GAME_DATE", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["GAME_DATE", "team", "pts_for", "pts_against"]

    away = g[["GAME_DATE", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["GAME_DATE", "team", "pts_for", "pts_against"]

    tg = pd.concat([home, away], ignore_index=True).sort_values(["team", "GAME_DATE"])

    # Rest days: days since *prior* game for that team
    tg["prev_date"] = tg.groupby("team")["GAME_DATE"].shift()
    tg["rest_days"] = (tg["GAME_DATE"] - tg["prev_date"]).dt.days

    # Rolling form from prior games (shift avoids leakage)
    tg["off_r10"] = (
        tg.groupby("team")["pts_for"]
          .transform(lambda s: s.shift().rolling(ROLL, min_periods=MIN_PERIODS).mean())
    )

    # Use a "defense strength" where HIGHER = BETTER defense.
    # Since pts_against lower is better, define def_str = -rolling_mean(pts_against).
    tg["def_r10_raw"] = tg.groupby("team")["pts_against"].transform(
        lambda s: s.shift().rolling(ROLL, min_periods=MIN_PERIODS).mean()
    )
    tg["def_str_r10"] = -tg["def_r10_raw"]

    # Join per-team rolling & rest back onto matchups
    h = tg.rename(columns={
        "team": "home_team",
        "off_r10": "home_off_r10",
        "def_str_r10": "home_def_r10",
        "rest_days": "home_rest",
    })[["GAME_DATE", "home_team", "home_off_r10", "home_def_r10", "home_rest"]]

    a = tg.rename(columns={
        "team": "away_team",
        "off_r10": "away_off_r10",
        "def_str_r10": "away_def_r10",
        "rest_days": "away_rest",
    })[["GAME_DATE", "away_team", "away_off_r10", "away_def_r10", "away_rest"]]

    g = (
        g.merge(h, on=["GAME_DATE", "home_team"], how="left")
         .merge(a, on=["GAME_DATE", "away_team"], how="left")
    )

    # Pregame deltas
    g["delta_off"] = g["home_off_r10"] - g["away_off_r10"]
    g["delta_def"] = g["home_def_r10"] - g["away_def_r10"]  # now positive = better defense for home
    g["delta_rest"] = g["home_rest"] - g["away_rest"]

    # Keep only rows with enough pregame history
    g = g.dropna(subset=["delta_off", "delta_def", "delta_rest"]).reset_index(drop=True)

    # Elo: compute PRE-game ratings and merge (no leakage)
    base_cols = ["GAME_DATE", "home_team", "home_score", "away_team", "away_score"]
    elo = add_elo(g[base_cols])
    g = elo.merge(g, on=base_cols, how="inner")
    g["delta_elo"] = g["home_elo_pre"] - g["away_elo_pre"]

    # Final selection
    feats = g[
        [
            "GAME_DATE",
            "home_team",
            "away_team",
            "delta_off",
            "delta_def",
            "delta_rest",
            "delta_elo",
            "home_win",
        ]
    ].sort_values("GAME_DATE").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(feats):,} rows -> {OUT_PATH}")

if __name__ == "__main__":
    build_features()
