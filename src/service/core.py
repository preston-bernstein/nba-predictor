from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.elo import add_elo

ROLL = 10
MINP = 3


def _last_rest_days(df: pd.DataFrame, team: str) -> int | None:
    t = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("GAME_DATE")
    if len(t) < 2:
        return None
    return int((t["GAME_DATE"].iloc[-1] - t["GAME_DATE"].iloc[-2]).days)


def _last_elo(df: pd.DataFrame, team: str) -> float | None:
    # compute pregame Elo then read the team's last pregame rating
    g = add_elo(df[["GAME_DATE", "home_team", "home_score", "away_team", "away_score"]])
    h = g[g["home_team"] == team][["home_elo_pre"]].tail(1)
    a = g[g["away_team"] == team][["away_elo_pre"]].tail(1)

    last_home = float(h["home_elo_pre"].iloc[-1]) if not h.empty else None
    last_away = float(a["away_elo_pre"].iloc[-1]) if not a.empty else None

    if last_home is not None:
        return last_home
    if last_away is not None:
        return last_away
    return None


def _team_form(df: pd.DataFrame, team: str) -> tuple[float, float] | None:
    home = df[["GAME_DATE", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["GAME_DATE", "team", "pts_for", "pts_against"]
    away = df[["GAME_DATE", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["GAME_DATE", "team", "pts_for", "pts_against"]
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg[tg["team"] == team].sort_values("GAME_DATE")
    # need at least MINP prior games for both rolling series
    if len(tg) < MINP + 1:
        return None
    off = tg["pts_for"].shift().rolling(ROLL, min_periods=MINP).mean().iloc[-1]
    deff = tg["pts_against"].shift().rolling(ROLL, min_periods=MINP).mean().iloc[-1]
    if np.isnan(off) or np.isnan(deff):
        return None
    return float(off), float(deff)


def compute_matchup_deltas(df: pd.DataFrame, home_team: str, away_team: str) -> dict[str, float]:
    """
    Pure domain logic: given a *pre-filtered* games dataframe (e.g., up to a date),
    compute matchup deltas for home vs away. Raises ValueError on bad input.
    """
    teams = set(df["home_team"]).union(df["away_team"])
    if home_team not in teams or away_team not in teams:
        raise ValueError("unknown team")

    h, a = _team_form(df, home_team), _team_form(df, away_team)
    if h is None or a is None:
        raise ValueError("insufficient history")

    (h_off, h_def), (a_off, a_def) = h, a
    deltas = {
        "delta_off": h_off - a_off,
        "delta_def": h_def - a_def,
    }

    # opportunistic extras; safe if unavailable
    hr, ar = _last_rest_days(df, home_team), _last_rest_days(df, away_team)
    if hr is not None and ar is not None:
        deltas["delta_rest"] = hr - ar

    he, ae = _last_elo(df, home_team), _last_elo(df, away_team)
    if he is not None and ae is not None:
        deltas["delta_elo"] = he - ae

    return deltas
