import pandas as pd

from src.data import features as features_mod
from src.data.features import _main as features_main


def test_build_features_writes_file_and_columns(tmp_path, monkeypatch):
    # ---- 1) make a tiny synthetic games.csv with enough history ----
    dates = pd.date_range("2024-10-01", periods=8, freq="D")
    rows = []
    for i, d in enumerate(dates):
        home, away = ("NYK", "BOS") if i % 2 == 0 else ("BOS", "NYK")
        home_score, away_score = 100 + i, 95 + i
        rows.append({
            "GAME_DATE": d,
            "home_team": home,
            "home_score": home_score,
            "away_team": away,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
        })
    games = pd.DataFrame(rows)

    in_path = tmp_path / "games.csv"
    out_path = tmp_path / "features.csv"
    games.to_csv(in_path, index=False)

    # ---- 2) point module paths to tmp (features.py exposes these) ----
    monkeypatch.setattr(features_mod, "IN_PATH", in_path, raising=True)
    monkeypatch.setattr(features_mod, "OUT_PATH", out_path, raising=True)

    # ---- 3) run feature builder (I/O wrapper) ----
    features_mod.build_features()

    # ---- 4) assertions (lightweight) ----
    assert out_path.exists(), "features.csv was not written"
    df = pd.read_csv(out_path, parse_dates=["GAME_DATE"])
    assert len(df) > 0

    expected_cols = {
        "GAME_DATE",
        "home_team",
        "away_team",
        "delta_off",
        "delta_def",
        "delta_rest",
        "delta_elo",
        "home_win",
    }
    assert expected_cols.issubset(df.columns)


def test__main_invokes_build_features(monkeypatch):
    called = {"ok": False}

    def fake_build_features():
        called["ok"] = True

    # Patch the name where it's looked up: inside src.data.features
    monkeypatch.setattr("src.data.features.build_features", fake_build_features, raising=True)

    features_main()
    assert called["ok"] is True
