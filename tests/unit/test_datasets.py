# tests/test_datasets.py
import pandas as pd
import numpy as np
from pathlib import Path

from src.model import datasets as ds

def _mini_df():
    # 8 rows with dates increasing; binary labels alternating
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range("2024-01-01", periods=8, freq="D"),
            "home_team": ["A","B","C","D","E","F","G","H"],
            "away_team": ["H","G","F","E","D","C","B","A"],
            "delta_off": [1,2,3,4,5,6,7,8],
            "delta_def": [0,-1,2,-3,4,-5,6,-7],
            "home_win":  [0,1,0,1,0,1,0,1],
        }
    )
    return df

def test_pick_features_ok():
    df = _mini_df()
    used = ds.pick_features(df, pref=["delta_off","delta_def","delta_rest"], min_feats=2)
    assert used == ["delta_off","delta_def"]

def test_pick_features_raises_when_too_few():
    df = _mini_df()[["GAME_DATE","home_team","away_team","home_win"]].copy()
    try:
        ds.pick_features(df, pref=["delta_off","delta_def"], min_feats=2)
    except ValueError as e:
        assert "Not enough features" in str(e)
    else:
        assert False, "expected ValueError"

def test_time_split_shape():
    df = _mini_df()
    tr, te = ds.time_split(df, test_frac=0.25)
    assert len(te) == 2  # 25% of 8 -> 2
    assert tr["GAME_DATE"].max() < te["GAME_DATE"].min()

def test_to_xy_types_and_shapes():
    df = _mini_df()
    used = ["delta_off","delta_def"]
    X, y = ds.to_xy(df, used)
    assert X.shape == (len(df), 2)
    assert set(np.unique(y)).issubset({0,1})

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
