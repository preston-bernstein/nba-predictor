from functools import lru_cache
from pathlib import Path
import joblib, numpy as np, pandas as pd

DATA = Path("data_cache/games.csv")
MODEL = Path("artifacts/model.joblib")
ROLL, MINP = 10, 3

@lru_cache(maxsize=1)
def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")
    need = {"GAME_DATE","home_team","home_score","away_team","away_score"}
    miss = need - set(df.columns)
    if miss: raise RuntimeError(f"games.csv missing: {sorted(miss)}")
    return df

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL)

def team_form(df: pd.DataFrame, team: str):
    home = df[["GAME_DATE","home_team","home_score","away_score"]].copy()
    home.columns = ["GAME_DATE","team","pts_for","pts_against"]
    away = df[["GAME_DATE","away_team","away_score","home_score"]].copy()
    away.columns = ["GAME_DATE","team","pts_for","pts_against"]
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg[tg["team"] == team].sort_values("GAME_DATE")
    off = tg["pts_for"].shift().rolling(ROLL, min_periods=MINP).mean().iloc[-1]
    deff = tg["pts_against"].shift().rolling(ROLL, min_periods=MINP).mean().iloc[-1]
    if np.isnan(off) or np.isnan(deff): return None
    return float(off), float(deff)

def matchup_features(home_team: str, away_team: str):
    df = load_games()
    teams = set(df["home_team"]).union(df["away_team"])
    if home_team not in teams or away_team not in teams:
        raise ValueError("unknown team")
    h, a = team_form(df, home_team), team_form(df, away_team)
    if h is None or a is None: raise ValueError("insufficient history")
    (h_off, h_def), (a_off, a_def) = h, a
    return h_off - a_off, h_def - a_def
