import joblib
import pandas as pd
from pytest import raises

from src.service import deps as deps_mod


def _mini_games_unsorted():
    # two rows out of order on purpose
    return pd.DataFrame([
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


def test_load_games_reads_sorts_and_caches(tmp_path, monkeypatch):
    # write minimal CSV
    csv = tmp_path / "games.csv"
    _mini_games_unsorted().to_csv(csv, index=False)

    # point config path used by deps + clear cache
    monkeypatch.setattr(deps_mod.config, "GAMES", csv, raising=True)
    deps_mod.load_games.cache_clear()

    # first call: loads and sorts
    df1 = deps_mod.load_games()
    assert df1["GAME_DATE"].is_monotonic_increasing

    # second call: uses lru_cache (same object identity)
    df2 = deps_mod.load_games()
    assert id(df1) == id(df2)


def test_load_games_through_filters_by_date(monkeypatch):
    df = _mini_games_unsorted()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # bypass disk: make load_games return df
    monkeypatch.setattr(deps_mod, "load_games", lambda: df, raising=True)

    filtered = deps_mod.load_games_through("2024-10-02")
    # strictly < date -> should include only 2024-10-01
    assert len(filtered) == 1
    assert filtered["GAME_DATE"].iloc[0].date().isoformat() == "2024-10-01"


def test_load_model_reads_and_caches(tmp_path, monkeypatch):
    deps_mod.load_model.cache_clear()
    obj = {"ok": True}
    model_path = tmp_path / "model.joblib"
    joblib.dump(obj, model_path)
    monkeypatch.setattr(deps_mod.config, "MODEL", model_path, raising=True)

    m1 = deps_mod.load_model()
    m2 = deps_mod.load_model()
    assert m1 == obj
    assert id(m1) == id(m2)  # cached


def test_matchup_features_wires_core_and_handles_return(monkeypatch):
    # stub the data source used inside matchup_features
    dummy_df = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2024-10-01"])})
    monkeypatch.setattr(deps_mod, "load_games_through", lambda date: dummy_df, raising=True)

    # capture the df passed into core and return fixed deltas
    seen = {}

    def fake_compute(df, home, away):
        seen["df_id"] = id(df)
        seen["args"] = (home, away)
        return {"delta_off": 1.2, "delta_def": -0.3, "delta_rest": 1, "delta_elo": 5}

    monkeypatch.setattr(deps_mod.core, "compute_matchup_deltas", fake_compute, raising=True)

    # return_dict=True returns whole mapping
    out_map = deps_mod.matchup_features("NYK", "BOS", date="2024-11-01", return_dict=True)
    assert out_map["delta_off"] == 1.2 and out_map["delta_def"] == -0.3
    assert seen["args"] == ("NYK", "BOS")

    # return_dict=False returns the pair
    d_off, d_def = deps_mod.matchup_features("NYK", "BOS")
    assert d_off == 1.2 and d_def == -0.3


def test_matchup_features_propagates_domain_errors(monkeypatch):
    # Make sure there are no games so deps hits the domain path
    monkeypatch.setattr(deps_mod, "load_games_through", lambda date: pd.DataFrame(), raising=True)

    # Force the core to raise the domain error we want to see bubble up
    def boom(df, h, a):
        raise ValueError("unknown team")

    monkeypatch.setattr(deps_mod.core, "compute_matchup_deltas", boom, raising=True)

    # Assert the error propagates (case-insensitive match on message)
    with raises(ValueError, match=r"(?i)unknown"):
        deps_mod.matchup_features("NYK", "???")
