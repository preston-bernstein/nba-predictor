import joblib
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.service import deps as deps_mod


def _mini_games_unsorted() -> pd.DataFrame:
    df = pd.DataFrame([
        dict(
            GAME_DATE="2024-10-02",
            home_team="BOS",
            home_score=101,
            away_team="NYK",
            away_score=99,
        ),
        dict(
            GAME_DATE="2024-10-01",
            home_team="NYK",
            home_score=100,
            away_team="BOS",
            away_score=98,
        ),
    ])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Ensure lru_cache state never leaks across tests."""
    deps_mod.load_games.cache_clear()
    deps_mod.load_model.cache_clear()
    if hasattr(deps_mod.load_games_through, "cache_clear"):
        deps_mod.load_games_through.cache_clear()


# ---------- tests ----------


def test_load_games_reads_sorts_and_caches(tmp_path, monkeypatch):
    csv = tmp_path / "games.csv"
    _mini_games_unsorted().to_csv(csv, index=False)

    monkeypatch.setattr(deps_mod.config, "GAMES", csv, raising=True)

    # First call should be a miss; second call a hit.
    info0 = deps_mod.load_games.cache_info()
    df1 = deps_mod.load_games()
    info1 = deps_mod.load_games.cache_info()
    assert df1["GAME_DATE"].is_monotonic_increasing
    assert info1.misses == info0.misses + 1

    df2 = deps_mod.load_games()
    info2 = deps_mod.load_games.cache_info()
    assert info2.hits == info1.hits + 1
    assert df1 is df2


def test_load_games_through_filters_by_date(monkeypatch):
    df = _mini_games_unsorted()
    monkeypatch.setattr(deps_mod, "load_games", lambda: df, raising=True)

    filtered = deps_mod.load_games_through("2024-10-02")
    assert len(filtered) == 1
    assert filtered["GAME_DATE"].iloc[0].date().isoformat() == "2024-10-01"


def test_load_games_through_none_returns_all(monkeypatch):
    df = _mini_games_unsorted()
    monkeypatch.setattr(deps_mod, "load_games", lambda: df, raising=True)

    result = deps_mod.load_games_through(None)
    assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))


def test_load_model_reads_and_caches(tmp_path, monkeypatch):
    obj = {"ok": True}
    model_path = tmp_path / "model.joblib"
    joblib.dump(obj, model_path)
    monkeypatch.setattr(deps_mod.config, "MODEL", model_path, raising=True)

    info0 = deps_mod.load_model.cache_info()
    m1 = deps_mod.load_model()
    info1 = deps_mod.load_model.cache_info()
    assert m1 == obj
    assert info1.misses == info0.misses + 1

    m2 = deps_mod.load_model()
    info2 = deps_mod.load_model.cache_info()
    assert m2 is m1
    assert info2.hits == info1.hits + 1


def test_matchup_features_wires_core_and_handles_return(monkeypatch):
    dummy_df = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2024-10-01"])})
    monkeypatch.setattr(deps_mod, "load_games_through", lambda date=None: dummy_df, raising=True)

    seen: dict[str, object] = {}

    def fake_compute(df, home, away):
        seen["df_id"] = id(df)
        seen["args"] = (home, away)
        return {"delta_off": 1.2, "delta_def": -0.3, "delta_rest": 1, "delta_elo": 5}

    monkeypatch.setattr(deps_mod.core, "compute_matchup_deltas", fake_compute, raising=True)

    out_map = deps_mod.matchup_features("NYK", "BOS", date="2024-11-01", return_dict=True)
    assert out_map["delta_off"] == 1.2 and out_map["delta_def"] == -0.3
    assert seen["args"] == ("NYK", "BOS")

    d_off, d_def = deps_mod.matchup_features("NYK", "BOS")
    assert (d_off, d_def) == (1.2, -0.3)


def test_matchup_features_propagates_domain_errors(monkeypatch):
    monkeypatch.setattr(
        deps_mod, "load_games_through", lambda date=None: pd.DataFrame(), raising=True
    )

    def boom(df, h, a):
        raise ValueError("unknown team")

    monkeypatch.setattr(deps_mod.core, "compute_matchup_deltas", boom, raising=True)

    with pytest.raises(ValueError, match=r"(?i)unknown"):
        deps_mod.matchup_features("NYK", "???")


@pytest.mark.parametrize(
    "inp,teams_in_data,normalize_team,canonical_name,expected,err",
    [
        ("NYK", set(), lambda t: "NYK", lambda c: "New York Knicks", "NYK", None),
        ("NYK", {"NYK", "BOS"}, lambda t: t, lambda c: "New York Knicks", "NYK", None),
        (
            "foo",
            {"NYK", "BOS"},
            lambda t: "LAL",
            lambda c: "Los Angeles Lakers",
            None,
            r"unknown team 'foo'",
        ),
    ],
)
def test__resolve_for_df_cases(
    inp, teams_in_data, normalize_team, canonical_name, expected, err, monkeypatch
):
    monkeypatch.setattr(deps_mod, "normalize_team", normalize_team, raising=True)
    monkeypatch.setattr(deps_mod, "canonical_name", canonical_name, raising=True)
    if err:
        with pytest.raises(ValueError, match=err):
            deps_mod._resolve_for_df(inp, teams_in_data)
    else:
        out = deps_mod._resolve_for_df(inp, teams_in_data)
        assert out == expected


def test_load_games_missing_columns_raises(tmp_path, monkeypatch):
    csv = tmp_path / "games.csv"
    pd.DataFrame([
        dict(GAME_DATE="2024-10-01", home_team="NYK", home_score=100, away_team="BOS"),
        dict(GAME_DATE="2024-10-02", home_team="BOS", home_score=101, away_team="NYK"),
    ]).to_csv(csv, index=False)

    monkeypatch.setattr(deps_mod.config, "GAMES", csv, raising=True)

    with pytest.raises(RuntimeError) as e:
        deps_mod.load_games()

    msg = str(e.value)
    assert "games.csv missing:" in msg
    assert "away_score" in msg


def test__teams_from_df_returns_union_of_labels():
    df = pd.DataFrame([
        {"home_team": "ATL", "away_team": "NYK"},
        {"home_team": "BOS", "away_team": "ATL"},
        {"home_team": "NYK", "away_team": "MIA"},
    ])

    got = deps_mod._teams_from_df(df)
    assert got == {"ATL", "NYK", "BOS", "MIA"}
