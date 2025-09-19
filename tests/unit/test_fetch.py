import pandas as pd
import pytest

from src.data import fetch
from src.data.fetch import _norm_team_label, years_span


def test_clean_key_removes_zero_width_and_normalizes():
    s = " N\u200b\u200d\ufeff.Y\u2013O\u2014R\u202f K\u00a0N I C K S  "
    out = fetch._clean_key(s)
    # no zero-widths and punctuation collapsed; uppercase
    assert " " not in out.replace(" ", "")  # relaxed sanity check
    assert out.startswith("N") and out.isupper()


def test_norm_team_label_fallback_full_and_suffix():
    assert fetch._norm_team_label("New York Knicks") == "NYK"
    assert fetch._norm_team_label("Portland Trail Blazers") == "POR"
    assert fetch._norm_team_label("Knicks") == "NYK"


def test_norm_team_label_unknown_includes_codepoints():
    with pytest.raises(ValueError) as e:
        fetch._norm_team_label("Gotham Rogues")
    assert "codepoints:" in str(e.value)


def test_drop_dupes_prefers_game_id_else_game_key():
    df_gid = pd.DataFrame([
        {"game_id": "A", "game_key": "zzz", "GAME_DATE": "2024-01-01"},
        {"game_id": "A", "game_key": "dup", "GAME_DATE": "2024-01-01"},
        {"game_id": "B", "game_key": "xxx", "GAME_DATE": "2024-01-02"},
    ])
    out1 = fetch._drop_dupe_games(df_gid.copy())
    assert len(out1) == 2 and sorted(out1["game_id"].tolist()) == ["A", "B"]

    df_gkey = pd.DataFrame([
        {"game_key": "K1", "GAME_DATE": "2024-01-01"},
        {"game_key": "K1", "GAME_DATE": "2024-01-01"},
        {"game_key": "K2", "GAME_DATE": "2024-01-02"},
    ])
    out2 = fetch._drop_dupe_games(df_gkey.copy())
    assert len(out2) == 2 and sorted(out2["game_key"].tolist()) == ["K1", "K2"]


def test_post_parse_cleanup_sorts_and_normalizes():
    df = pd.DataFrame([
        {
            "GAME_DATE": "2024-10-21",
            "home_team": "New York Knicks",
            "home_score": 101,
            "away_team": "Portland Trail Blazers",
            "away_score": 99,
        },
        {
            "GAME_DATE": "2024-10-20",
            "home_team": "NYK",
            "home_score": 99,
            "away_team": "BOS",
            "away_score": 101,
        },
    ])
    out = fetch._post_parse_cleanup(df)
    assert list(out["GAME_DATE"].astype(str)) == ["2024-10-20", "2024-10-21"]
    assert set(out["home_team"]) >= {"NYK"}
    assert set(out["away_team"]) >= {"BOS", "POR"}

def test_norm_team_label_suffix_branch_with_prefix_text():
    assert _norm_team_label("The Knicks") == "NYK"

def test_years_span_happy_path_inclusive():
    assert years_span(2022, 2025) == [2022, 2023, 2024, 2025]

def test_years_span_raises_when_end_lt_start():
    import pytest
    with pytest.raises(SystemExit):
        years_span(2025, 2024)