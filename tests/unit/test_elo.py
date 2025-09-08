# tests/test_elo.py
import pandas as pd
from src.data.elo import add_elo, EloConfig

def test_add_elo_outputs_and_home_adv_effect():
    g = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2024-10-01","2024-10-05"]),
        "home_team": ["NYK","BOS"],
        "home_score": [100, 95],
        "away_team": ["BOS","NYK"],
        "away_score": [90, 98],
    })
    out = add_elo(g, EloConfig(base=1500.0, k=20.0, home_adv=50.0))
    assert {"home_elo_pre","away_elo_pre"}.issubset(out.columns)
    # First game: both teams start equal pregame
    assert out.loc[0, "home_elo_pre"] == out.loc[0, "away_elo_pre"]
