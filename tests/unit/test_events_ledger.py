import os
import sqlite3

import pytest
from events_helpers import FakePredict, make_event, to_bytes
from fastapi.testclient import TestClient

from src.events.handler import handle_message
from src.events.ledger import Ledger, ledger_path, read_summary
from src.service.app import app


def put(ledger, game_id, prob, final=None):
    handle_message(ledger, FakePredict(prob=prob), to_bytes(make_event(game_id=game_id)))
    if final:
        handle_message(
            ledger,
            FakePredict(),
            to_bytes(make_event(game_id=game_id, status_kind="FINAL", score=final)),
        )


def test_ledger_path_default_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PREDICTION_LEDGER_PATH", raising=False)
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    assert ledger_path() == tmp_path / "prediction_ledger.sqlite"
    monkeypatch.setenv("PREDICTION_LEDGER_PATH", str(tmp_path / "a.db"))
    assert ledger_path() == tmp_path / "a.db"


def test_creates_parent_dir(tmp_path):
    Ledger(tmp_path / "deep" / "dir" / "l.sqlite")
    assert (tmp_path / "deep" / "dir" / "l.sqlite").exists()


def test_summary_empty_when_no_file(tmp_path):
    assert read_summary(tmp_path / "nope.sqlite") == {
        "games": 0,
        "settled": 0,
        "graded": 0,
        "accuracy": None,
    }
    assert not (tmp_path / "nope.sqlite").exists()


def test_summary_counts_and_accuracy(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite")
    put(ledger, "a", 0.8, final=(100, 90))  # predicted home, home won: right
    put(ledger, "b", 0.3, final=(100, 90))  # predicted away, home won: wrong
    put(ledger, "c", 0.4, final=(90, 100))  # predicted away, away won: right
    put(ledger, "d", 0.9)  # not settled
    handle_message(  # settled, no prediction: not graded
        ledger, FakePredict(), to_bytes(make_event(game_id="e", status_kind="FINAL", score=(1, 2)))
    )
    s = read_summary(tmp_path / "l.sqlite")
    assert s == {"games": 5, "settled": 4, "graded": 3, "accuracy": pytest.approx(2 / 3)}


def test_summary_accuracy_null_without_graded(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite")
    put(ledger, "d", 0.9)
    s = read_summary(tmp_path / "l.sqlite")
    assert (s["games"], s["settled"], s["graded"], s["accuracy"]) == (1, 0, 0, None)


def test_summary_is_read_only(tmp_path):
    Ledger(tmp_path / "l.sqlite")
    before = os.stat(tmp_path / "l.sqlite").st_mtime_ns
    read_summary(tmp_path / "l.sqlite")
    assert os.stat(tmp_path / "l.sqlite").st_mtime_ns == before
    with pytest.raises(sqlite3.OperationalError):
        # the endpoint opens read-only, so writes must fail
        from src.events.ledger import _connect_ro

        _connect_ro(tmp_path / "l.sqlite").execute("DELETE FROM games")


def test_summary_endpoint(monkeypatch, tmp_path):
    path = tmp_path / "api.sqlite"
    ledger = Ledger(path)
    put(ledger, "a", 0.8, final=(100, 90))
    monkeypatch.setenv("PREDICTION_LEDGER_PATH", str(path))
    r = TestClient(app).get("/v1/ledger/summary")
    assert r.status_code == 200
    assert r.json() == {"games": 1, "settled": 1, "graded": 1, "accuracy": 1.0}


def test_summary_endpoint_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTION_LEDGER_PATH", str(tmp_path / "none.sqlite"))
    r = TestClient(app).get("/v1/ledger/summary")
    assert r.json() == {"games": 0, "settled": 0, "graded": 0, "accuracy": None}
