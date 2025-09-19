from __future__ import annotations

from typing import Final

import pandas as pd

# NEW: use the service normalizer so train-time matches serve-time
from src.service.normalizer import TeamNormalizeError, normalize_team

from .elo import add_elo

ROLL: Final[int] = 10
MINP: Final[int] = 3


def _canonize_team_cols(g: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize team identifiers to canonical 3-letter codes (e.g., 'NYK', 'BOS').
    Fail loudly on unknown teams so we don't bake bad rows into training.
    """

    def _norm(v: str) -> str:
        try:
            return normalize_team(v)
        except TeamNormalizeError as err:
            raise ValueError(f"Unknown team in input games: {v!r}") from err

    g["home_team"] = g["home_team"].map(_norm)
    g["away_team"] = g["away_team"].map(_norm)
    return g


def team_game_rows(g: pd.DataFrame) -> pd.DataFrame:
    """Expand game rows into team-game rows (one per team per game)."""
    home = g[["GAME_DATE", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["GAME_DATE", "team", "pts_for", "pts_against"]
    away = g[["GAME_DATE", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["GAME_DATE", "team", "pts_for", "pts_against"]
    tg = pd.concat([home, away], ignore_index=True)
    return tg.sort_values(["team", "GAME_DATE"])


def add_rest_days(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.copy()
    tg["prev_date"] = tg.groupby("team")["GAME_DATE"].shift()
    tg["rest_days"] = (tg["GAME_DATE"] - tg["prev_date"]).dt.days
    return tg


def rolling_form(tg: pd.DataFrame, roll: int = ROLL, minp: int = MINP) -> pd.DataFrame:
    """Attach rolling offensive/defensive form from *prior* games (shift to avoid leakage)."""
    tg = tg.copy()
    tg["off_r10"] = tg.groupby("team")["pts_for"].transform(
        lambda s: s.shift().rolling(roll, min_periods=minp).mean()
    )
    tg["def_r10"] = tg.groupby("team")["pts_against"].transform(
        lambda s: s.shift().rolling(roll, min_periods=minp).mean()
    )
    return tg


def join_matchups(g: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Join team-game features back onto the original matchup rows."""
    h = tg.rename(
        columns={
            "team": "home_team",
            "off_r10": "home_off_r10",
            "def_r10": "home_def_r10",
            "rest_days": "home_rest",
        }
    )[["GAME_DATE", "home_team", "home_off_r10", "home_def_r10", "home_rest"]]

    a = tg.rename(
        columns={
            "team": "away_team",
            "off_r10": "away_off_r10",
            "def_r10": "away_def_r10",
            "rest_days": "away_rest",
        }
    )[["GAME_DATE", "away_team", "away_off_r10", "away_def_r10", "away_rest"]]

    g2 = g.sort_values("GAME_DATE").merge(h, on=["GAME_DATE", "home_team"], how="left")
    g2 = g2.merge(a, on=["GAME_DATE", "away_team"], how="left")
    return g2


def add_pregame_deltas(gm: pd.DataFrame) -> pd.DataFrame:
    gm = gm.copy()
    gm["delta_off"] = gm["home_off_r10"] - gm["away_off_r10"]
    gm["delta_def"] = gm["home_def_r10"] - gm["away_def_r10"]
    gm["delta_rest"] = gm["home_rest"] - gm["away_rest"]
    # keep rows with enough history
    return gm.dropna(subset=["delta_off", "delta_def", "delta_rest"]).reset_index(drop=True)


def merge_elo_features(gm: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo pregame ratings and attach delta_elo."""
    base_cols = ["GAME_DATE", "home_team", "home_score", "away_team", "away_score"]
    elo_df = add_elo(gm[base_cols])
    g3 = elo_df.merge(gm, on=base_cols, how="inner")
    g3["delta_elo"] = g3["home_elo_pre"] - g3["away_elo_pre"]
    return g3


def build_features_df(games: pd.DataFrame) -> pd.DataFrame:
    """Pure function: games -> features dataframe (no file I/O).
    Input teams can be codes or names; we normalize to canonical codes once here.
    """
    # NEW: normalize team IDs up front to prevent train/serve drift
    games = _canonize_team_cols(games)

    tg = team_game_rows(games)
    tg = add_rest_days(tg)
    tg = rolling_form(tg)

    gm = join_matchups(games, tg)
    gm = add_pregame_deltas(gm)
    gm = merge_elo_features(gm)

    feats = (
        gm[
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
        ]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    return feats
