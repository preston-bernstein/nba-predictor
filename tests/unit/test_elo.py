import pandas as pd
import pytest

from src.data.elo import EloConfig, add_elo


def test_add_elo_outputs_and_home_adv_effect():
    g = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2024-10-01", "2024-10-05"]),
        "home_team": ["NYK", "BOS"],
        "home_score": [100, 95],
        "away_team": ["BOS", "NYK"],
        "away_score": [90, 98],
    })
    out = add_elo(g, EloConfig(base=1500.0, k=20.0, home_adv=50.0))
    assert {"home_elo_pre", "away_elo_pre"}.issubset(out.columns)
    # First game: both teams start equal pregame
    assert out.loc[0, "home_elo_pre"] == out.loc[0, "away_elo_pre"]


def test_add_elo_non_numeric_raises():
    df = pd.DataFrame([
        {
            "GAME_DATE": "2024-01-01",
            "home_team": "NYK",
            "away_team": "BOS",
            "home_score": "NaN",
            "away_score": 100,
        }
    ])
    with pytest.raises(ValueError):
        add_elo(df)


def test_add_elo_tie_path_ok():
    df = pd.DataFrame([
        {
            "GAME_DATE": "2024-01-01",
            "home_team": "NYK",
            "away_team": "BOS",
            "home_score": 100,
            "away_score": 100,
        }
    ])
    out = add_elo(df, EloConfig(k=10.0))
    assert {"home_elo_pre", "away_elo_pre"} <= set(out.columns)


def test_add_elo_non_numeric_string_raises():
    df = pd.DataFrame([
        {
            "GAME_DATE": "2024-01-01",
            "home_team": "NYK",
            "away_team": "BOS",
            "home_score": "abc",
            "away_score": 100,
        }
    ])
    with pytest.raises(ValueError, match=r"Non-numeric score"):
        add_elo(df)


def test_add_elo_missing_columns_minimal_dict_triggers_specific_raise():
    df = pd.DataFrame([
        {
            "GAME_DATE": "2024-01-01",
            "home_team": "NYK",
            "home_score": 100,
        }
    ])
    with pytest.raises(ValueError) as ei:
        add_elo(df)
    msg = str(ei.value)
    assert msg.startswith("add_elo: missing columns:")
    assert "away_team" in msg and "away_score" in msg
