import pandas as pd
from src.service import core as core_mod

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
    # optional keys may or may not be there; if present, they’re floats
    for k in ("delta_rest", "delta_elo"):
        if k in deltas:
            assert isinstance(deltas[k], float) or isinstance(deltas[k], int)

def test_compute_matchup_deltas_unknown_team():
    games = make_games()
    try:
        core_mod.compute_matchup_deltas(games, "NYK", "???")
        assert False, "expected ValueError for unknown team"
    except ValueError as e:
        assert "unknown" in str(e)

def test_compute_matchup_deltas_insufficient_history():
    # too few games to satisfy rolling min_periods
    short = make_games().iloc[:3].copy()
    try:
        core_mod.compute_matchup_deltas(short, "NYK", "BOS")
        assert False, "expected ValueError for insufficient history"
    except ValueError as e:
        assert "insufficient" in str(e)
