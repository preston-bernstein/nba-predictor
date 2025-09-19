import pandas as pd

from src.data.transform import build_features_df


def _mini_games():
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
    return pd.DataFrame(rows)


def test_build_features_df_shapes_and_columns():
    games = _mini_games()
    feats = build_features_df(games)

    assert len(feats) > 0
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
    assert expected_cols.issubset(feats.columns)


def test_build_features_df_no_nans_and_types():
    games = _mini_games()
    feats = build_features_df(games)

    for col in ["delta_off", "delta_def", "delta_rest", "delta_elo"]:
        assert feats[col].notna().all(), f"{col} has NaNs"
        assert pd.api.types.is_numeric_dtype(feats[col]), f"{col} not numeric"

    assert feats["GAME_DATE"].is_monotonic_increasing
