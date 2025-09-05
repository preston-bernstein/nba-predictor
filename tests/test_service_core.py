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
            dict(GAME_DATE=d, home_team=home, home_score=home_score,
                 away_team=away, away_score=away_score)
        )
    return pd.DataFrame(rows)

def test_team_form_and_matchup_features(monkeypatch):
    games = make_games()

    # team_form can be called directly
    nyk = core_mod.team_form(games, "NYK")
    bos = core_mod.team_form(games, "BOS")
    assert nyk is not None and bos is not None
    off, deff = nyk
    assert isinstance(off, float) and isinstance(deff, float)

    # matchup_features uses load_games -> stub it
    monkeypatch.setattr(core_mod, "load_games", lambda: games, raising=True)
    d_off, d_def = core_mod.matchup_features("NYK", "BOS")
    assert isinstance(d_off, float) and isinstance(d_def, float)

def test_matchup_features_unknown(monkeypatch):
    games = make_games()
    monkeypatch.setattr(core_mod, "load_games", lambda: games, raising=True)
    try:
        core_mod.matchup_features("NYK", "???")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown" in str(e)
