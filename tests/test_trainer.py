# tests/test_trainer.py
import pandas as pd
from pathlib import Path

from src.model.trainer import Trainer

def _mini_features_df():
    # small but stratifiable set (ensures ROC-AUC defined)
    return pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                ["2024-10-20","2024-10-21","2024-10-22","2024-10-23",
                 "2024-10-24","2024-10-25","2024-10-26","2024-10-27"]
            ),
            "home_team": ["NYK","GSW","BOS","LAL","MIA","CHI","DAL","DEN"],
            "away_team": ["BOS","LAL","NYK","GSW","CHI","MIA","DEN","DAL"],
            "delta_off": [ 5, 3,-4,-2, 6,-5, 2,-1],
            "delta_def": [-1,2, 3,-2,-3, 4, 0, 1],
            "delta_rest":[ 1,0, 2, 1,-1, 0, 3,-2],
            "delta_elo": [10,5,-8,-3,12,-6, 4,-2],
            "home_win":  [ 1, 1, 0, 0, 1, 0, 1, 0],
        }
    )

def test_trainer_runs_and_persists_best(tmp_path: Path):
    feats = tmp_path / "features.csv"
    _mini_features_df().to_csv(feats, index=False)

    art = tmp_path / "artifacts"

    trainer = Trainer(
        feats_path=feats,
        art_dir=art,
        pref_features=("delta_off", "delta_def", "delta_rest", "delta_elo"),
        min_features=2,
        test_frac=0.25,
    )

    metrics = trainer.run(model_names=["logreg", "rf"])

    # artifacts
    assert (art / "metrics.json").exists()
    assert (art / "model.joblib").exists()

    # best model file was also produced (source of the copy)
    best = metrics["best_model"]
    assert (art / f"model-{best}.joblib").exists()

    # schema / ranges
    for key in ["n_train", "n_test", "accuracy", "roc_auc"]:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # runs for both models recorded
    assert set(metrics["runs"].keys()) == {"logreg", "rf"}

def test_trainer_respects_feature_preferences(tmp_path: Path):
    # Only provide two preferred features; trainer should still work
    df = _mini_features_df()[[
        "GAME_DATE","home_team","away_team",
        "delta_off","delta_def","home_win"
    ]]
    feats = tmp_path / "features.csv"
    df.to_csv(feats, index=False)
    art = tmp_path / "artifacts"

    trainer = Trainer(
        feats_path=feats,
        art_dir=art,
        pref_features=("delta_off", "delta_def", "delta_rest", "delta_elo"),
        min_features=2,
        test_frac=0.25,
    )
    metrics = trainer.run(model_names=["logreg"])

    # features_used should reflect only available columns
    assert metrics["features_used"] == ["delta_off", "delta_def"]
