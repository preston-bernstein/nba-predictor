from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from src.service.app import app
from src.service import routes as routes_mod

class DummyModel:
    def predict_proba(self, X):
        # deterministic: sigmoid of sum of features -> two-class proba
        s = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.column_stack([1 - s, s])

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"ok": True}

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
    r = client.get("/teams")
    assert r.status_code == 200
    assert set(r.json()) == {"NYK", "BOS"}

def test_predict_ok(monkeypatch):
    # stub features + model
    monkeypatch.setattr(
        routes_mod, "matchup_features",
        lambda h, a, **kw: (1.5, -0.5),   # 👈 accept date= (and future kwargs)
        raising=True
    )
    monkeypatch.setattr(routes_mod, "load_model", lambda: DummyModel(), raising=True)

    client = TestClient(app)
    r = client.get("/predict", params={"home": "NYK", "away": "BOS", "date": "2024-11-01"})  # optional: exercise date
    assert r.status_code == 200
    body = r.json()
    assert body["home_team"] == "NYK" and body["away_team"] == "BOS"
    assert 0 <= body["prob_home_win"] <= 1

def test_predict_bad_input(monkeypatch):
    # make matchup_features raise same error core would for unknown team
    def boom(h, a, **kw):   # 👈 accept date=
        raise ValueError("unknown team")
    monkeypatch.setattr(routes_mod, "matchup_features", boom, raising=True)
    monkeypatch.setattr(routes_mod, "load_model", lambda: DummyModel(), raising=True)

    client = TestClient(app)
    r = client.get("/predict", params={"home": "NYK", "away": "??", "date": "2024-11-01"})  # optional: pass date
    assert r.status_code == 422
    assert "unknown" in r.json()["detail"].lower()
