import pandas as pd
from fastapi.testclient import TestClient

from src.service import routes as routes_mod
from src.service.app import app

API_PREFIX = "/v1"


class DummyModel:
    # emulate scikit API; expect 2 features (off, def)
    n_features_in_ = 2

    def predict_proba(self, X):
        import numpy as np

        # simple deterministic mapping so the test can assert shape
        p = 1 / (1 + np.exp(-X[:, 0] * 0.1 + X[:, 1] * 0.05))
        return np.vstack([1 - p, p]).T


def test_teams(monkeypatch):
    df = pd.DataFrame(
        dict(
            GAME_DATE=pd.date_range("2024-10-01", periods=2),
            home_team=["NYK", "BOS"],
            home_score=[100, 101],
            away_team=["BOS", "NYK"],
            away_score=[99, 98],
        )
    )
    monkeypatch.setattr(routes_mod, "load_games", lambda: df, raising=True)

    client = TestClient(app)
    r = client.get(f"{API_PREFIX}/teams")
    assert r.status_code == 200
    assert set(r.json()["teams"]) == {"NYK", "BOS"}


def test_predict_ok(monkeypatch):
    # stub features -> DICT (canonical now)
    monkeypatch.setattr(
        routes_mod,
        "matchup_features",
        lambda h, a, **kw: {
            "delta_off": 1.5,
            "delta_def": -0.5,
            "delta_rest": 0.0,
            "delta_elo": 2.0,
        },
        raising=True,
    )
    # stub model
    monkeypatch.setattr(routes_mod, "load_model", lambda: DummyModel(), raising=True)

    client = TestClient(app)
    r = client.get(
        f"{API_PREFIX}/predict", params={"home": "NYK", "away": "BOS", "date": "2024-11-01"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["home_team"] == "NYK" and body["away_team"] == "BOS"

    # ✅ The API may include extra features; ensure at least those required by the model exist.
    assert {"delta_off", "delta_def"}.issubset(body["features"].keys())

    # Optional: sanity check probability is a float in [0,1]
    assert isinstance(body["prob_home_win"], float)
    assert 0.0 <= body["prob_home_win"] <= 1.0


def test_predict_bad_input(monkeypatch):
    # make matchup_features raise domain error (propagates as 422)
    def boom(h, a, **kw):
        raise ValueError("unknown team")

    monkeypatch.setattr(routes_mod, "matchup_features", boom, raising=True)

    client = TestClient(app)
    r = client.get(f"{API_PREFIX}/predict", params={"home": "NYK", "away": "???"})
    assert r.status_code == 422
    assert "unknown" in r.json()["detail"]
