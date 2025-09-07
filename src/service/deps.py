from __future__ import annotations
from functools import lru_cache
import pandas as pd, joblib
from typing import Iterable, Set

from src import config
from . import core

@lru_cache(maxsize=1)
def load_games() -> pd.DataFrame:
    # read path dynamically so monkeypatching config.GAMES works
    df = pd.read_csv(config.GAMES, parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")
    need = {"GAME_DATE", "home_team", "home_score", "away_team", "away_score"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"games.csv missing: {sorted(miss)}")
    return df

@lru_cache(maxsize=1)
def load_model():
    # read path dynamically so monkeypatching config.MODEL works
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

def _normalize_team(team: str, teams_in_data: Iterable[str] | None = None) -> str:
    if not team or not teams_in_data:
        return team
    aliases = {
        "NYK": "New York Knicks",
        "BOS": "Boston Celtics",
        "LAL": "Los Angeles Lakers",
        "GSW": "Golden State Warriors",
    }
    candidate = aliases.get(team, team)
    return candidate if candidate in set(teams_in_data) else team

def matchup_features(home: str, away: str, date: str | None = None, *, return_dict: bool = False):
    df = load_games_through(date)
    teams = _teams_from_df(df)
    home = _normalize_team(home, teams)
    away = _normalize_team(away, teams)

    deltas = core.compute_matchup_deltas(df, home, away)  # may raise ValueError

    if return_dict:
        return {k: float(v) for k, v in deltas.items()}

    return (float(deltas["delta_off"]), float(deltas["delta_def"]))
