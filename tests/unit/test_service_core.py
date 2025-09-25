import numbers

import pandas as pd
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
