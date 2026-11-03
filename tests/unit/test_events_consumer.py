import threading

import pytest
from events_helpers import (
    FakeConsumer,
    FakeMessage,
    FakePredict,
    FakeProducer,
    make_event,
    to_bytes,
)

from src.events import consumer as consumer_mod
from src.events.consumer import RetriesExhausted, Worker
from src.events.ledger import Ledger


def build(tmp_path, predict=None, producer=None, consumer=None, **kw):
    sleeps: list[float] = []
    ledger = Ledger(tmp_path / "l.sqlite")
    consumer = consumer or FakeConsumer()
    producer = producer or FakeProducer()
    worker = Worker(
        consumer,
        producer,
        ledger,
        predict or FakePredict(),
        sleep=sleeps.append,
        **kw,
    )
    return worker, ledger, consumer, producer, sleeps


def test_success_commits_after_write(tmp_path):
    worker, ledger, consumer, producer, _ = build(tmp_path)
    worker.process(FakeMessage(to_bytes(make_event()), partition=2, offset=11))
    assert ledger.get_game("g-1") is not None
    assert consumer.committed == [("game.updated", 2, 11)]
    assert producer.sent == []


def test_duplicate_is_committed_and_skipped(tmp_path):
    worker, ledger, consumer, _, _ = build(tmp_path)
    ev = to_bytes(make_event(event_id="e1"))
    worker.process(FakeMessage(ev, offset=1))
    worker.process(FakeMessage(ev, offset=2))
    assert ledger.count_events() == 1
    assert [c[2] for c in consumer.committed] == [1, 2]


@pytest.mark.parametrize("value", [b"{bad", to_bytes({"eventId": "x"}), None])
def test_poison_goes_to_dlq_with_headers_then_commit(tmp_path, value):
    worker, ledger, consumer, producer, _ = build(tmp_path, group_id="predictor")
    worker.process(FakeMessage(value, key=b"k1", partition=2, offset=42))
    assert len(producer.sent) == 1
    sent = producer.sent[0]
    assert sent["topic"] == "game.updated.dlq"
    assert sent["key"] == b"k1"
    assert sent["value"] == value
    h = sent["headers"]
    assert h["dlq-reason"]
    assert h["dlq-source-topic"] == "game.updated"
    assert h["dlq-source-partition"] == "2"
    assert h["dlq-source-offset"] == "42"
    assert h["dlq-consumer-group"] == "predictor"
    assert consumer.committed == [("game.updated", 2, 42)]
    assert ledger.count_events() == 0


def test_dlq_failure_does_not_commit_and_retries(tmp_path):
    class Flaky(FakeProducer):
        def produce(self, *a, **k):
            if len(self.sent) < 2:
                self.sent.append({})  # count attempts
                k["on_delivery"]("broker down", None)
            else:
                super().produce(*a, **k)

    worker, _, consumer, producer, sleeps = build(tmp_path, producer=Flaky())
    worker.process(FakeMessage(b"{bad"))
    assert sleeps == [0.5, 1.0]
    assert consumer.committed != []


def test_dlq_unflushed_counts_as_failure(tmp_path):
    worker, _, consumer, _, sleeps = build(
        tmp_path, producer=FakeProducer(unflushed=1), max_retries=2
    )
    with pytest.raises(RetriesExhausted):
        worker.process(FakeMessage(b"{bad"))
    assert consumer.committed == []
    assert len(sleeps) == 2


def test_transient_error_retries_with_backoff_then_commits(tmp_path):
    attempts = []

    def predict(game):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("model missing")
        return FakePredict()(game)

    worker, ledger, consumer, _, sleeps = build(tmp_path, predict=predict)
    worker.process(FakeMessage(to_bytes(make_event())))
    assert sleeps == [0.5, 1.0]
    assert len(consumer.committed) == 1
    assert ledger.get_game("g-1") is not None


def test_backoff_is_capped(tmp_path):
    def predict(game):
        raise OSError("x")

    worker, _, consumer, _, sleeps = build(
        tmp_path, predict=predict, max_retries=5, base_delay=1.0, max_delay=3.0
    )
    with pytest.raises(RetriesExhausted):
        worker.process(FakeMessage(to_bytes(make_event())))
    assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0]
    assert consumer.committed == []


def test_run_polls_until_stopped_and_skips_errors(tmp_path):
    stop = threading.Event()
    msgs = [
        FakeMessage(None, error="some kafka error"),
        FakeMessage(to_bytes(make_event(game_id="a")), offset=1),
        FakeMessage(to_bytes(make_event(game_id="b")), offset=2),
    ]
    consumer = FakeConsumer(msgs)
    consumer.on_empty = stop.set
    worker, ledger, consumer, _, _ = build(tmp_path, consumer=consumer)
    worker.run(stop)
    assert consumer.subscribed == ["game.updated"]
    assert [c[2] for c in consumer.committed] == [1, 2]
    assert ledger.get_game("a") and ledger.get_game("b")
    assert consumer.closed


def test_run_closes_consumer_when_retries_exhausted(tmp_path):
    def predict(game):
        raise OSError("x")

    consumer = FakeConsumer([FakeMessage(to_bytes(make_event()))])
    worker, _, consumer, _, _ = build(tmp_path, predict=predict, consumer=consumer, max_retries=1)
    with pytest.raises(RetriesExhausted):
        worker.run(threading.Event())
    assert consumer.closed
    assert consumer.committed == []


def test_run_stops_during_backoff_without_commit(tmp_path):
    stop = threading.Event()

    def predict(game):
        stop.set()
        raise OSError("x")

    consumer = FakeConsumer([FakeMessage(to_bytes(make_event()))])
    worker, _, consumer, _, _ = build(tmp_path, predict=predict, consumer=consumer)
    worker.run(stop)
    assert consumer.committed == []
    assert consumer.closed


# ---- main() and config wiring ----


def test_main_disabled_without_brokers(monkeypatch, caplog):
    monkeypatch.delenv("KAFKA_BROKERS", raising=False)
    assert consumer_mod.main() == 0


def test_client_configs(monkeypatch):
    monkeypatch.setenv("KAFKA_GROUP_ID", "g9")
    cfg = consumer_mod.consumer_config("b:9092")
    assert cfg["group.id"] == "g9"
    assert cfg["enable.auto.commit"] is False
    assert cfg["auto.offset.reset"] == "earliest"
    assert cfg["bootstrap.servers"] == "b:9092"
    assert consumer_mod.producer_config("b:9092")["bootstrap.servers"] == "b:9092"


def test_main_wires_everything(monkeypatch, tmp_path):
    created = {}

    class C(FakeConsumer):
        def __init__(self, cfg):
            super().__init__([FakeMessage(to_bytes(make_event()), offset=3)])
            created["consumer_cfg"] = cfg
            self.on_empty = lambda: created["stop"].set()

    class P(FakeProducer):
        def __init__(self, cfg):
            super().__init__()
            created["producer_cfg"] = cfg

    monkeypatch.setenv("KAFKA_BROKERS", "kafka:9092")
    monkeypatch.setenv("PREDICTION_LEDGER_PATH", str(tmp_path / "x" / "l.sqlite"))
    monkeypatch.setenv("KAFKA_TOPIC", "t1")
    monkeypatch.setenv("KAFKA_DLQ_TOPIC", "t1.dlq")
    monkeypatch.setattr(consumer_mod, "Consumer", C)
    monkeypatch.setattr(consumer_mod, "Producer", P)
    monkeypatch.setattr(consumer_mod, "predict_game", FakePredict())
    monkeypatch.setattr(consumer_mod, "setup_logging", lambda: None)
    real_event = threading.Event

    def make_stop():
        created["stop"] = real_event()
        return created["stop"]

    monkeypatch.setattr(consumer_mod.threading, "Event", make_stop)
    assert consumer_mod.main() == 0
    assert created["consumer_cfg"]["bootstrap.servers"] == "kafka:9092"
    assert Ledger(tmp_path / "x" / "l.sqlite").get_game("g-1") is not None


def test_main_returns_1_when_retries_exhausted(monkeypatch, tmp_path):
    class C(FakeConsumer):
        def __init__(self, cfg):
            super().__init__([FakeMessage(to_bytes(make_event()))])

    def boom(game):
        raise OSError("x")

    monkeypatch.setenv("KAFKA_BROKERS", "kafka:9092")
    monkeypatch.setenv("PREDICTION_LEDGER_PATH", str(tmp_path / "l.sqlite"))
    monkeypatch.setenv("KAFKA_MAX_RETRIES", "1")
    monkeypatch.setattr(consumer_mod, "Consumer", C)
    monkeypatch.setattr(consumer_mod, "Producer", lambda cfg: FakeProducer())
    monkeypatch.setattr(consumer_mod, "predict_game", boom)
    monkeypatch.setattr(consumer_mod, "setup_logging", lambda: None)
    monkeypatch.setattr(consumer_mod.time, "sleep", lambda s: None)
    assert consumer_mod.main() == 1
