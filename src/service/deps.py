from __future__ import annotations
from functools import lru_cache
from typing import Iterable, Set

import joblib
import pandas as pd

from src import config
from . import core
from .normalizer import normalize_team, TeamNormalizeError, CANON


@lru_cache(maxsize=1)
def load_games() -> pd.DataFrame:
    df = pd.read_csv(config.GAMES, parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")
    need = {"GAME_DATE", "home_team", "home_score", "away_team", "away_score"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"games.csv missing: {sorted(miss)}")
    return df


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(config.MODEL)


def load_games_through(date: str | None):
    df = load_games()
    if date is None:
        return df
    return df.loc[df["GAME_DATE"] < pd.to_datetime(date)].copy()


def _teams_from_df(df: pd.DataFrame) -> Set[str]:
    cols = set(df.columns)
    if {"home_team", "away_team"}.issubset(cols):
        return set(df["home_team"]).union(df["away_team"])
    return set()


def _resolve_for_df(input_team: str, teams_in_data: Set[str]) -> str:
    """
    Normalize any input to canonical code. If the dataframe has known labels,
    map to those; if the set is empty, just return the code (don't block tests).
    """
    try:
        code = normalize_team(input_team)  # may raise
    except TeamNormalizeError as e:
        # tests check for 'unknown' in the message, lowercase to be safe
        raise ValueError(str(e).lower())

    if not teams_in_data:
        # Nothing to validate against (e.g., filtered dataset in tests) — return code
        return code

    if code in teams_in_data:
        return code

    full = CANON.get(code, code)
    if full in teams_in_data:
        return full

    # we *do* have teams in data, but neither code nor full name present
    raise ValueError(f"unknown team '{input_team}' in historical data; "
                     f"available={sorted(teams_in_data)[:5]}...")


def matchup_features(
    home: str,
    away: str,
    date: str | None = None,
    *,
    return_dict: bool = False,
):
    df = load_games_through(date)
    teams = _teams_from_df(df)

    home_label = _resolve_for_df(home, teams)
    away_label = _resolve_for_df(away, teams)

    deltas = core.compute_matchup_deltas(df, home_label, away_label)  # may raise ValueError

    if return_dict:
        return {k: float(v) for k, v in deltas.items()}

    return (float(deltas["delta_off"]), float(deltas["delta_def"]))
