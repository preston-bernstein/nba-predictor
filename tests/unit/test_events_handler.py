import json
import sqlite3

import pytest
from events_helpers import FakePredict, make_event, to_bytes

from src.events.handler import PoisonMessage, handle_message
from src.events.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.sqlite")


def test_scheduled_stores_prediction(ledger):
    predict = FakePredict(prob=0.64, version="m1")
    assert handle_message(ledger, predict, to_bytes(make_event())) == "applied"
    row = ledger.get_game("g-1")
    assert row["home_team"] == "New York Knicks"
    assert row["away_team"] == "Boston Celtics"
    assert row["start_time"] == "2025-01-05T00:30:00Z"
    assert row["prob_home_win"] == 0.64
    assert row["model_version"] == "m1"
    assert row["settled"] == 0
    assert row["outcome"] is None
    assert row["home_score"] is None


def test_prediction_uses_game_start_date(ledger):
    predict = FakePredict()
    handle_message(ledger, predict, to_bytes(make_event(start_time="2025-01-05T00:30:00Z")))
    assert predict.calls[0]["startDate"] == "2025-01-05"
    assert predict.calls[0]["homeTeam"]["name"] == "New York Knicks"


def test_in_progress_also_predicts(ledger):
    predict = FakePredict()
    handle_message(ledger, predict, to_bytes(make_event(status_kind="IN_PROGRESS", score=(10, 8))))
    assert len(predict.calls) == 1
    assert ledger.get_game("g-1")["prob_home_win"] == 0.7


def test_final_settles_and_keeps_prediction(ledger):
    predict = FakePredict(prob=0.6)
    handle_message(ledger, predict, to_bytes(make_event()))
    handle_message(ledger, predict, to_bytes(make_event(status_kind="FINAL", score=(100, 90))))
    row = ledger.get_game("g-1")
    assert (row["home_score"], row["away_score"]) == (100, 90)
    assert row["outcome"] == "home"
    assert row["settled"] == 1
    assert row["prob_home_win"] == 0.6
    assert len(predict.calls) == 1


def test_final_away_win(ledger):
    handle_message(ledger, FakePredict(), to_bytes(make_event(status_kind="FINAL", score=(80, 90))))
    assert ledger.get_game("g-1")["outcome"] == "away"


def test_final_without_prior_prediction_stores_null(ledger):
    predict = FakePredict()
    handle_message(ledger, predict, to_bytes(make_event(status_kind="FINAL", score=(100, 90))))
    row = ledger.get_game("g-1")
    assert row["prob_home_win"] is None
    assert row["model_version"] is None
    assert row["settled"] == 1
    assert predict.calls == []


def test_duplicate_event_id_applied_once(ledger):
    predict = FakePredict()
    ev = to_bytes(make_event(event_id="e-1"))
    assert handle_message(ledger, predict, ev) == "applied"
    assert handle_message(ledger, predict, ev) == "duplicate"
    assert len(predict.calls) == 1
    assert ledger.count_events() == 1


def test_duplicate_final_not_reapplied(ledger):
    ev = to_bytes(make_event(status_kind="FINAL", score=(100, 90), event_id="e-f"))
    handle_message(ledger, FakePredict(), ev)
    # Same id, different scores: still skipped.
    changed = to_bytes(make_event(status_kind="FINAL", score=(1, 2), event_id="e-f"))
    assert handle_message(ledger, FakePredict(), changed) == "duplicate"
    assert ledger.get_game("g-1")["home_score"] == 100


def test_prediction_never_overwritten(ledger):
    handle_message(ledger, FakePredict(prob=0.7, version="a"), to_bytes(make_event()))
    handle_message(
        ledger, FakePredict(prob=0.2, version="b"), to_bytes(make_event(status_kind="IN_PROGRESS"))
    )
    row = ledger.get_game("g-1")
    assert (row["prob_home_win"], row["model_version"]) == (0.7, "a")


def test_scheduled_after_final_does_not_add_prediction(ledger):
    predict = FakePredict()
    handle_message(ledger, predict, to_bytes(make_event(status_kind="FINAL", score=(100, 90))))
    handle_message(ledger, predict, to_bytes(make_event()))
    assert ledger.get_game("g-1")["prob_home_win"] is None
    assert predict.calls == []


def test_postponed_and_canceled_are_acknowledged_only(ledger):
    for kind in ("POSTPONED", "CANCELED"):
        assert handle_message(ledger, FakePredict(), to_bytes(make_event(status_kind=kind))) == (
            "ignored"
        )
    assert ledger.get_game("g-1") is None
    assert ledger.count_events() == 2


def test_failed_write_leaves_no_event_row(ledger, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(ledger, "_upsert_prediction", boom)
    with pytest.raises(sqlite3.OperationalError):
        handle_message(ledger, FakePredict(), to_bytes(make_event(event_id="e-x")))
    assert ledger.count_events() == 0
    assert ledger.get_game("g-1") is None


@pytest.mark.parametrize(
    "value,reason",
    [
        (None, "empty"),
        (b"{not json", "invalid JSON"),
        (b"\xff\xfe", "invalid JSON"),
        (b"[1]", "schema"),
        (json.dumps({"eventId": "x"}).encode(), "schema"),
    ],
)
def test_poison_inputs(ledger, value, reason):
    with pytest.raises(PoisonMessage) as exc:
        handle_message(ledger, FakePredict(), value)
    assert reason in exc.value.reason


def test_schema_mismatch_wrong_version(ledger):
    ev = make_event()
    ev["schemaVersion"] = 2
    with pytest.raises(PoisonMessage) as exc:
        handle_message(ledger, FakePredict(), to_bytes(ev))
    assert "schema" in exc.value.reason


def test_final_tie_is_poison(ledger):
    with pytest.raises(PoisonMessage) as exc:
        handle_message(
            ledger, FakePredict(), to_bytes(make_event(status_kind="FINAL", score=(90, 90)))
        )
    assert "tie" in exc.value.reason


def test_unresolvable_team_is_poison(ledger):
    def predict(game):
        raise PoisonMessage("unresolvable team: nope")

    with pytest.raises(PoisonMessage):
        handle_message(ledger, predict, to_bytes(make_event()))
    assert ledger.get_game("g-1") is None
    assert ledger.count_events() == 0


def test_transient_predict_error_propagates(ledger):
    def predict(game):
        raise OSError("model file missing")

    with pytest.raises(OSError):
        handle_message(ledger, predict, to_bytes(make_event()))
    assert ledger.count_events() == 0
