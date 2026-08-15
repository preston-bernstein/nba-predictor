from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.service.errors import register_handlers


def _make_app(raises: Exception) -> FastAPI:
    app = FastAPI()
    register_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise raises

    return app


def test_value_error_handler_returns_422_and_logs(caplog):
    client = TestClient(_make_app(ValueError("bad domain input")))

    with caplog.at_level("WARNING", logger="src.service.errors"):
        r = client.get("/boom")

    assert r.status_code == 422
    assert r.json() == {"detail": "bad domain input"}
    assert any("ValueError handled as 422" in rec.message for rec in caplog.records)


def test_key_error_handler_returns_400_and_logs(caplog):
    client = TestClient(_make_app(KeyError("team_code")))

    with caplog.at_level("WARNING", logger="src.service.errors"):
        r = client.get("/boom")

    assert r.status_code == 400
    assert "team_code" in r.json()["detail"]
    assert any("KeyError handled as 400" in rec.message for rec in caplog.records)
