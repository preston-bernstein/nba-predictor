"""Shared fakes and sample events for the Kafka consumer tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from src.events.handler import Prediction

SCHEMA_PATH = Path(__file__).resolve().parent / "fixtures" / "game.updated.v1.json"


def make_event(
    game_id: str = "g-1",
    status_kind: str = "SCHEDULED",
    score: tuple[int, int] = (0, 0),
    **extra: Any,
) -> dict:
    """A game.updated event. extra may set home, away, event_id and start_time."""
    home = extra.get("home", ("NYK", "New York Knicks"))
    away = extra.get("away", ("BOS", "Boston Celtics"))
    return {
        "eventId": extra.get("event_id") or str(uuid.uuid4()),
        "eventType": "game.updated",
        "schemaVersion": 1,
        "occurredAt": "2025-01-04T12:00:00Z",
        "changeKind": "created",
        "game": {
            "id": game_id,
            "provider": "test",
            "homeTeam": {"id": home[0], "name": home[1]},
            "awayTeam": {"id": away[0], "name": away[1]},
            "startTime": extra.get("start_time", "2025-01-05T00:30:00Z"),
            "status": status_kind.title(),
            "statusKind": status_kind,
            "score": {"home": score[0], "away": score[1]},
        },
    }


def to_bytes(event: dict) -> bytes:
    return json.dumps(event).encode()


class FakePredict:
    """Stands in for the prediction core. Records every call."""

    def __init__(self, prob: float = 0.7, version: str = "v-test") -> None:
        self.prob = prob
        self.version = version
        self.calls: list[dict] = []

    def __call__(self, game: dict) -> Prediction:
        self.calls.append(game)
        return Prediction(self.prob, self.version)


class FakeMessage:
    """A Kafka message. fields may set key, topic, partition, offset and error."""

    def __init__(self, value: bytes | None, **fields: Any) -> None:
        self._value = value
        self._key = fields.get("key", b"g-1")
        self._topic = fields.get("topic", "game.updated")
        self._partition = fields.get("partition", 1)
        self._offset = fields.get("offset", 7)
        self._error = fields.get("error")

    def value(self) -> bytes | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def error(self) -> object:
        return self._error


class FakeProducer:
    def __init__(self, fail_delivery: bool = False, unflushed: int = 0) -> None:
        self.sent: list[dict] = []
        self.fail_delivery = fail_delivery
        self.unflushed = unflushed

    def produce(self, topic, **kw):
        self.sent.append({
            "topic": topic,
            "key": kw.get("key"),
            "value": kw.get("value"),
            "headers": dict(kw.get("headers") or []),
        })
        on_delivery = kw.get("on_delivery")
        if on_delivery:
            on_delivery("boom" if self.fail_delivery else None, None)

    def flush(self, timeout=None) -> int:
        return self.unflushed


class FakeConsumer:
    def __init__(self, messages: list | None = None) -> None:
        self.messages = list(messages or [])
        self.committed: list[tuple[str, int, int]] = []
        self.subscribed: list[str] = []
        self.closed = False
        self.on_empty = None

    def subscribe(self, topics):
        self.subscribed = list(topics)

    def poll(self, timeout=None):
        if self.messages:
            return self.messages.pop(0)
        if self.on_empty:
            self.on_empty()
        return None

    def commit(self, message=None, asynchronous=True):
        assert asynchronous is False
        self.committed.append((message.topic(), message.partition(), message.offset()))

    def close(self):
        self.closed = True
