import json

import pandas as pd

from src import config as config_mod
from src.model import train as train_mod


def test_train_multi_models(tmp_path):
    # tiny but stratifiable set
    df = pd.DataFrame({
        "GAME_DATE": pd.to_datetime([
            "2024-10-20",
            "2024-10-21",
            "2024-10-22",
            "2024-10-23",
            "2024-10-24",
            "2024-10-25",
            "2024-10-26",
            "2024-10-27",
        ]),
        "home_team": ["NYK", "GSW", "BOS", "LAL", "MIA", "CHI", "DAL", "DEN"],
        "away_team": ["BOS", "LAL", "NYK", "GSW", "CHI", "MIA", "DEN", "DAL"],
        "delta_off": [5, 3, -4, -2, 6, -5, 2, -1],
        "delta_def": [-1, 2, 3, -2, -3, 4, 0, 1],
        "delta_rest": [1, 0, 2, 1, -1, 0, 3, -2],
        "delta_elo": [10, 5, -8, -3, 12, -6, 4, -2],
        "home_win": [1, 1, 0, 0, 1, 0, 1, 0],
    })
    feats = tmp_path / "features.csv"
    df.to_csv(feats, index=False)

    # point config paths to tmp
    config_mod.FEATS = feats
    config_mod.ART_DIR = tmp_path / "artifacts"

    # run with 2 models
    train_mod.main(models=["logreg", "rf"])

    # verify artifacts and metrics
    metrics_path = config_mod.ART_DIR / "metrics.json"
    model_path = config_mod.ART_DIR / "model.joblib"
    assert metrics_path.exists() and model_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["best_model"] in {"logreg", "rf"}
    assert set(metrics["runs"].keys()) == {"logreg", "rf"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
