import json
import pandas as pd
from pathlib import Path

from src.model import train as train_mod

def test_train_writes_artifacts(tmp_path, monkeypatch):
    # 1) Create a tiny fake features.csv
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                ["2024-10-20","2024-10-21","2024-10-22","2024-10-23","2024-10-24","2024-10-25"]
            ),
            "home_team": ["NYK","GSW","BOS","LAL","MIA","CHI"],
            "away_team": ["BOS","LAL","NYK","GSW","CHI","MIA"],
            # simple separable features
            "delta_off": [ 5,  3, -4, -2,  6, -5],
            "delta_def": [-1,  2,  3, -2, -3,  4],
            "home_win":  [ 1,  1,  0,  0,  1,  0],
        }
    )
    feats_path = tmp_path / "features.csv"
    df.to_csv(feats_path, index=False)

    # 2) Point the module to tmp paths
    monkeypatch.setattr(train_mod, "FEATS", feats_path, raising=True)
    monkeypatch.setattr(train_mod, "ART", tmp_path / "artifacts", raising=True)

    # 3) Run training
    train_mod.main()

    # 4) Check artifacts
    model_path = train_mod.ART / "model.joblib"
    metrics_path = train_mod.ART / "metrics.json"
    assert model_path.exists()
    assert metrics_path.exists()

    # 5) Check metrics schema + ranges
    metrics = json.loads(metrics_path.read_text())
    for key in ["n_train", "n_test", "accuracy", "roc_auc"]:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
