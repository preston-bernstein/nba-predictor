import pandas as pd
from pathlib import Path
from src.data import features as features_mod

def test_build_features_writes_file_and_columns(tmp_path, monkeypatch):
    # ---- 1) make a tiny synthetic games.csv with enough history ----
    dates = pd.date_range("2024-10-01", periods=8, freq="D")
    rows = []
    # NYK vs BOS alternating home; plenty of history for rolling=10 with min_periods=3
    for i, d in enumerate(dates):
        if i % 2 == 0:
            home, away = "NYK", "BOS"
        else:
            home, away = "BOS", "NYK"
        home_score = 100 + i
        away_score = 95 + i
        rows.append(
            {
                "GAME_DATE": d,
                "home_team": home,
                "home_score": home_score,
                "away_team": away,
                "away_score": away_score,
                "home_win": int(home_score > away_score),
            }
        )
    games = pd.DataFrame(rows)

    in_path = tmp_path / "games.csv"
    out_path = tmp_path / "features.csv"
    games.to_csv(in_path, index=False)

    # ---- 2) point module paths to tmp ----
    monkeypatch.setattr(features_mod, "IN_PATH", in_path, raising=True)
    monkeypatch.setattr(features_mod, "OUT_PATH", out_path, raising=True)

    # ---- 3) run feature builder ----
    features_mod.build_features()

    # ---- 4) assertions ----
    assert out_path.exists(), "features.csv was not written"

    df = pd.read_csv(out_path, parse_dates=["GAME_DATE"])
    # Should have at least a few rows after min_periods=3 filter
    assert len(df) > 0

    # required columns present
    expected_cols = {"GAME_DATE", "home_team", "away_team", "delta_off", "delta_def", "home_win"}
    assert expected_cols.issubset(df.columns)

    # no NaNs in the delta columns after filtering
    assert df["delta_off"].notna().all()
    assert df["delta_def"].notna().all()

    # dates sorted ascending (your code sorts by GAME_DATE before writing)
    assert df["GAME_DATE"].is_monotonic_increasing
