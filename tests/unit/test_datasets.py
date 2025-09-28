import numpy as np
import pandas as pd
import pytest

from src.model import datasets as ds
from src.model.datasets import load_features, time_split


def _mini_df():
    # 8 rows with dates increasing; binary labels alternating
    df = pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-01-01", periods=8, freq="D"),
        "home_team": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "away_team": ["H", "G", "F", "E", "D", "C", "B", "A"],
        "delta_off": [1, 2, 3, 4, 5, 6, 7, 8],
        "delta_def": [0, -1, 2, -3, 4, -5, 6, -7],
        "home_win": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    return df


def test_pick_features_ok():
    df = _mini_df()
    used = ds.pick_features(df, pref=["delta_off", "delta_def", "delta_rest"], min_feats=2)
    assert used == ["delta_off", "delta_def"]


def test_pick_features_raises_when_too_few():
    df = _mini_df()[["GAME_DATE", "home_team", "away_team", "home_win"]].copy()

    with pytest.raises(ValueError) as exc:
        ds.pick_features(df, pref=["delta_off", "delta_def"], min_feats=2)
    assert "Not enough features" in str(exc.value)


def test_time_split_shape():
    df = _mini_df()
    tr, te = ds.time_split(df, test_frac=0.25)
    assert len(te) == 2  # 25% of 8 -> 2
    assert tr["GAME_DATE"].max() < te["GAME_DATE"].min()


def test_to_xy_types_and_shapes():
    df = _mini_df()
    used = ["delta_off", "delta_def"]
    X, y = ds.to_xy(df, used)
    assert X.shape == (len(df), 2)
    assert set(np.unique(y)).issubset({0, 1})


def test_baseline_stats_keys():
    df = _mini_df()
    base = ds.baseline_stats(df)
    assert "baseline_home_rate" in base and "baseline_home_acc" in base
    assert 0.0 <= base["baseline_home_rate"] <= 1.0
    assert base["baseline_home_rate"] == base["baseline_home_acc"]


def test_load_features_roundtrip(tmp_path):
    # write a minimal features.csv to disk and load via util
    p = tmp_path / "features.csv"
    df = _mini_df()
    df.to_csv(p, index=False)
    loaded = ds.load_features(p)
    assert len(loaded) == len(df)
    assert pd.api.types.is_datetime64_any_dtype(loaded["GAME_DATE"])


def test_load_features_missing_required_cols(tmp_path):
    p = tmp_path / "features.csv"
    df = pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-01-01", periods=3, freq="D"),
        "delta_off": [1, 2, 3],
    })
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match=r"^features file missing required columns:"):
        load_features(p)


def test_time_split_adjust_when_test_frac_ge_n():
    df = pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-01-01", periods=2, freq="D"),
        "home_win": [0, 1],
        "delta_off": [1, 2],
        "delta_def": [3, 4],
    })
    train, test = time_split(df, test_frac=1.0)
    assert len(train) == 1 and len(test) == 1
    assert train["GAME_DATE"].max() < test["GAME_DATE"].min()
