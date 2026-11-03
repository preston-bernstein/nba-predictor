import pytest
from fastapi import HTTPException

from src.events import predictions
from src.events.handler import PoisonMessage
from src.service.schemas import PredictResponse


def game(home_name="New York Knicks", home_id="NYK", away_name="Boston Celtics", away_id="BOS"):
    return {
        "homeTeam": {"id": home_id, "name": home_name},
        "awayTeam": {"id": away_id, "name": away_name},
        "startDate": "2025-01-05",
    }


def ok_response(q):
    return PredictResponse(
        home_team=q.home,
        away_team=q.away,
        as_of=q.date,
        features={"delta_off": 1.0, "delta_def": 1.0},
        prob_home_win=0.55,
    )


def test_uses_service_predict_with_team_name_and_date(monkeypatch, tmp_path):
    seen = []

    def fake(q):
        seen.append(q)
        return ok_response(q)

    model = tmp_path / "m.joblib"
    model.write_bytes(b"abc")
    monkeypatch.setattr(predictions.routes, "predict", fake)
    monkeypatch.setattr(predictions.config, "MODEL", model)
    predictions.model_version.cache_clear()

    p = predictions.predict_game(game())
    assert p.prob_home_win == 0.55
    assert (seen[0].home, seen[0].away, seen[0].date) == (
        "New York Knicks",
        "Boston Celtics",
        "2025-01-05",
    )
    assert p.model_version.startswith("m.joblib@")
    assert len(p.model_version.split("@")[1]) == 12


def test_falls_back_to_team_id(monkeypatch):
    calls = []

    def fake(q):
        calls.append(q.home)
        if q.home == "Weird Name":
            raise HTTPException(422, "unknown team")
        return ok_response(q)

    monkeypatch.setattr(predictions.routes, "predict", fake)
    monkeypatch.setattr(predictions, "model_version", lambda: "v")
    p = predictions.predict_game(game(home_name="Weird Name"))
    assert calls == ["Weird Name", "NYK"]
    assert p.prob_home_win == 0.55


def test_unresolvable_team_is_poison(monkeypatch):
    def fake(q):
        raise HTTPException(422, "unknown team 'x'")

    monkeypatch.setattr(predictions.routes, "predict", fake)
    with pytest.raises(PoisonMessage) as exc:
        predictions.predict_game(game())
    assert "unknown team" in exc.value.reason


def test_other_http_errors_are_not_swallowed(monkeypatch):
    def fake(q):
        raise HTTPException(500, "x")

    monkeypatch.setattr(predictions.routes, "predict", fake)
    with pytest.raises(HTTPException):
        predictions.predict_game(game())


def test_missing_model_is_transient_error(monkeypatch, tmp_path):
    monkeypatch.setattr(predictions.config, "MODEL", tmp_path / "none.joblib")
    predictions.model_version.cache_clear()
    with pytest.raises(OSError):
        predictions.model_version()


def test_runs_real_route_function_with_stubbed_data(monkeypatch):
    """No fake of routes.predict: only its data and model inputs are stubbed."""
    import numpy as np

    from src.service import routes

    class M:
        n_features_in_ = 2

        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    monkeypatch.setattr(
        routes, "matchup_features", lambda h, a, date=None: {"delta_off": 1.0, "delta_def": 2.0}
    )
    monkeypatch.setattr(routes, "load_model", lambda: M())
    monkeypatch.setattr(predictions, "model_version", lambda: "v")
    p = predictions.predict_game(game())
    assert p.prob_home_win == pytest.approx(0.6)
    assert p.model_version == "v"
