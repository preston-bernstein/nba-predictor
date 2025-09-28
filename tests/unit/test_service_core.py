import numbers

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from src.service import core as core_mod
from src.service.core import compute_matchup_deltas


def make_games():
    # minimal data with enough history (min_periods=3) for both teams
    rows = []
    dates = pd.date_range("2024-10-01", periods=8, freq="D")
    for i, d in enumerate(dates):
        if i % 2 == 0:
            home, away = "NYK", "BOS"
        else:
            home, away = "BOS", "NYK"
        home_score, away_score = 100 + i, 95 + i
        rows.append(
            dict(
                GAME_DATE=d,
                home_team=home,
                home_score=home_score,
                away_team=away,
                away_score=away_score,
            )
        )
    return pd.DataFrame(rows)


def test_compute_matchup_deltas_ok():
    games = make_games()
    deltas = core_mod.compute_matchup_deltas(games, "NYK", "BOS")
    # required keys always present
    assert "delta_off" in deltas and "delta_def" in deltas
    assert isinstance(deltas["delta_off"], float)
    assert isinstance(deltas["delta_def"], float)
    # optional keys may or may not be there; if present, they’re numbers.Real
    for k in ("delta_rest", "delta_elo"):
        if k in deltas:
            assert isinstance(deltas[k], numbers.Real)


def test_compute_matchup_deltas_unknown_team():
    games = make_games()
    # (?i) makes the match case-insensitive
    with raises(ValueError, match=r"(?i)unknown"):
        core_mod.compute_matchup_deltas(games, "NYK", "???")


def test_compute_matchup_deltas_insufficient_history():
    # too few games to satisfy rolling min_periods
    short = make_games().iloc[:3].copy()
    with raises(ValueError) as e:
        core_mod.compute_matchup_deltas(short, "NYK", "BOS")
    assert "insufficient" in str(e.value).lower()


def _make_games_with_spacing():
    rows = []
    dates = pd.to_datetime([
        "2024-10-01",
        "2024-10-03",
        "2024-10-05",
        "2024-10-08",
        "2024-10-10",
        "2024-10-12",
        "2024-10-15",
        "2024-10-18",
    ])
    for i, d in enumerate(dates):
        home, away = ("NYK", "BOS") if i % 2 == 0 else ("BOS", "NYK")
        rows.append({
            "GAME_DATE": d,
            "home_team": home,
            "home_score": 100 + i,
            "away_team": away,
            "away_score": 90 + i,
        })
    return pd.DataFrame(rows)


def test_compute_matchup_deltas_includes_optional_fields():
    games = _make_games_with_spacing()
    deltas = compute_matchup_deltas(games, "NYK", "BOS")
    assert {"delta_off", "delta_def"} <= set(deltas)
    assert "delta_rest" in deltas and "delta_elo" in deltas


@pytest.mark.parametrize("force_none", [True, False])
def test_compute_matchup_deltas_optional_fields_presence(force_none, monkeypatch):
    games = _make_games_with_spacing()

    if force_none:
        monkeypatch.setattr(core_mod, "_last_rest_days", lambda *_: None, raising=True)
        monkeypatch.setattr(core_mod, "_last_elo", lambda *_: None, raising=True)

    deltas = compute_matchup_deltas(games, "NYK", "BOS")
    assert {"delta_off", "delta_def"} <= set(deltas)
    if force_none:
        assert "delta_rest" not in deltas
        assert "delta_elo" not in deltas
    else:
        assert "delta_rest" in deltas
        assert "delta_elo" in deltas


def test__last_rest_days_single_game_returns_none():
    df = pd.DataFrame([
        dict(
            GAME_DATE=pd.Timestamp("2024-10-01"),
            home_team="NYK",
            home_score=100,
            away_team="BOS",
            away_score=99,
        )
    ])
    assert core_mod._last_rest_days(df, "NYK") is None
    assert core_mod._last_rest_days(df, "BOS") is None


def test__team_form_nan_returns_none():
    rows = [
        ("2024-10-01", "NYK", 100, "BOS", 98),
        ("2024-10-02", "NYK", np.nan, "PHI", 88),
        ("2024-10-03", "MIA", 90, "NYK", 92),
        ("2024-10-05", "NYK", 105, "ATL", 101),
    ]
    df = pd.DataFrame(
        rows, columns=["GAME_DATE", "home_team", "home_score", "away_team", "away_score"]
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    assert core_mod._team_form(df, "NYK") is None


def test__last_elo_away_branch_returns_value():
    df = pd.DataFrame(
        [
            ("2024-10-01", "BOS", 100, "NYK", 99),
            ("2024-10-03", "PHI", 95, "NYK", 90),
        ],
        columns=["GAME_DATE", "home_team", "home_score", "away_team", "away_score"],
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    val = core_mod._last_elo(df, "NYK")
    assert isinstance(val, float)


def test__last_elo_returns_none_when_team_absent():
    df = pd.DataFrame(
        [
            ("2024-10-01", "NYK", 100, "BOS", 98),
            ("2024-10-03", "BOS", 99, "NYK", 101),
        ],
        columns=["GAME_DATE", "home_team", "home_score", "away_team", "away_score"],
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    assert core_mod._last_elo(df, "LAL") is None
