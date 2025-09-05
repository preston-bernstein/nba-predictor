# tests/test_train_multi.py
import json
import pandas as pd
from pathlib import Path
from src.model import train as train_mod

def test_train_multi_models(tmp_path, monkeypatch):
    # tiny but stratifiable set
    df = pd.DataFrame(
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
    feats = tmp_path / "features.csv"
    df.to_csv(feats, index=False)

    # point paths to tmp
    monkeypatch.setattr(train_mod, "FEATS", feats, raising=True)
    monkeypatch.setattr(train_mod, "ART", tmp_path / "artifacts", raising=True)

    # run with 2 models
    train_mod.main(models=["logreg", "rf"])

    # artifacts
    art = train_mod.ART
    assert (art / "model-logreg.joblib").exists()
    assert (art / "model-rf.joblib").exists()
    assert (art / "model.joblib").exists()  # best

    # metrics schema
    metrics = json.loads((art / "metrics.json").read_text())
    assert "features_used" in metrics and "runs" in metrics
    assert set(metrics["runs"].keys()) == {"logreg", "rf"}
    for m in metrics["runs"].values():
        # accuracy range
        assert 0.0 <= m["accuracy"] <= 1.0
        # roc can be NaN on tiny folds; allow but type must exist
        assert "roc_auc" in m
