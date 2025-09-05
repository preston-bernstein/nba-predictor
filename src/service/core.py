from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd, joblib
from src.data.elo import add_elo  # ← you were missing this import

DATA = Path("data_cache/games.csv")
MODEL = Path("artifacts/model.joblib")
ROLL, MINP = 10, 3

@lru_cache(maxsize=1)
def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")
    need = {"GAME_DATE","home_team","home_score","away_team","away_score"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"games.csv missing: {sorted(miss)}")
    return df

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL)

def load_games_through(date: str | None):
    df = load_games()
    if date is None:
        return df
    return df.loc[df["GAME_DATE"] < pd.to_datetime(date)].copy()

def _last_rest_days(df: pd.DataFrame, team: str) -> int | None:
    t = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("GAME_DATE")
    if len(t) < 2:
        return None
    return int((t["GAME_DATE"].iloc[-1] - t["GAME_DATE"].iloc[-2]).days)

def _last_elo(df: pd.DataFrame, team: str) -> float | None:
    g = add_elo(df[["GAME_DATE","home_team","home_score","away_team","away_score"]])
    h = g[g["home_team"] == team][["home_elo_pre"]].tail(1)
    a = g[g["away_team"] == team][["away_elo_pre"]].tail(1)
    if not h.empty: return float(h["home_elo_pre"].iloc[0])
    if not a.empty: return float(a["away_elo_pre"].iloc[0])
    return None

def team_form(df: pd.DataFrame, team: str):
    home = df[["GAME_DATE","home_team","home_score","away_score"]].copy()
    home.columns = ["GAME_DATE","team","pts_for","pts_against"]
    away = df[["GAME_DATE","away_team","away_score","home_score"]].copy()
    away.columns = ["GAME_DATE","team","pts_for","pts_against"]
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

def matchup_features(home_team: str, away_team: str, date: str | None = None, *, return_dict: bool = False):
    df = load_games_through(date)
    teams = set(df["home_team"]).union(df["away_team"])
    if home_team not in teams or away_team not in teams:
        raise ValueError("unknown team")
    h, a = team_form(df, home_team), team_form(df, away_team)
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

    return deltas if return_dict else (deltas["delta_off"], deltas["delta_def"])
