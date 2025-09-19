import pandas as pd
import pytest

from src.service.deps import _resolve_for_df, load_games_through


def test_resolve_returns_code_when_present():
    teams = {"NYK", "BOS"}
    assert _resolve_for_df("NYK", teams) == "NYK"


def test_resolve_maps_full_name_if_present():
    teams = {"NEW YORK KNICKS", "BOSTON CELTICS"}
    assert _resolve_for_df("nyk", teams) == "NEW YORK KNICKS"


def test_resolve_errors_if_unknown_in_data():
    teams = {"NYK", "BOS"}
    with pytest.raises(ValueError):
        _resolve_for_df("LAL", teams)


def test_resolve_returns_code_when_no_teams_in_data(monkeypatch):
    # Patch names inside the module under test via dotted string
    monkeypatch.setattr("src.service.deps._teams_from_df", lambda df: set(), raising=True)
    monkeypatch.setattr(
        "src.service.deps.load_games_through", lambda date: pd.DataFrame(), raising=True
    )
    assert _resolve_for_df("nyk", set()) == "NYK"


def test_load_games_through_filters_by_date(monkeypatch):
    df = pd.DataFrame([
        {
            "GAME_DATE": pd.to_datetime("2024-10-01"),
            "home_team": "NYK",
            "home_score": 1,
            "away_team": "BOS",
            "away_score": 0,
        },
        {
            "GAME_DATE": pd.to_datetime("2024-10-10"),
            "home_team": "BOS",
            "home_score": 1,
            "away_team": "NYK",
            "away_score": 0,
        },
    ])
    monkeypatch.setattr("src.service.deps.load_games", lambda: df, raising=True)
    out = load_games_through("2024-10-05")
    assert len(out) == 1
    assert out["GAME_DATE"].max() < pd.to_datetime("2024-10-05")
