"""Dataset utilities: loading, splitting, and feature selection."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

REQUIRED_COLS = {"GAME_DATE", "home_win"}

def load_features(path: Path | str) -> pd.DataFrame:
    """Read features CSV and return time-sorted DataFrame with basic schema checks."""
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"features file missing required columns: {sorted(missing)}")
    return df.sort_values("GAME_DATE").reset_index(drop=True)

def pick_features(df: pd.DataFrame, pref: list[str], min_feats: int = 2) -> list[str]:
    """
    Return the ordered subset of `pref` that exist in df.
    Raise if fewer than `min_feats` are present.
    """
    used = [c for c in pref if c in df.columns]
    if len(used) < min_feats:
        raise ValueError(
            f"Not enough features. Found {used}, need ≥{min_feats} among {pref}"
        )
    return used

def time_split(df: pd.DataFrame, test_frac: float = 0.25, min_test: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological split: last `test_frac` as test (at least `min_test`)."""
    n = len(df)
    n_test = max(min_test, int(round(test_frac * n)))
    if n_test >= n:
        n_test = max(1, n - 1)  # keep at least 1 train row
    return df.iloc[:-n_test], df.iloc[-n_test:]

def to_xy(df: pd.DataFrame, features: list[str], target: str = "home_win") -> tuple[np.ndarray, np.ndarray]:
    """Convert a frame to (X, y) numpy arrays."""
    X = df[features].values
    y = df[target].values
    return X, y

def baseline_stats(test_df: pd.DataFrame) -> dict:
    """Compute simple test-fold baselines."""
    rate = float(test_df["home_win"].mean())
    return {"baseline_home_rate": rate, "baseline_home_acc": rate}
