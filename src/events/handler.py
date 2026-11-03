"""Turn one `game.updated` message into a ledger write."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .ledger import Ledger

SCHEMA_FILE = Path(__file__).resolve().parent / "game.updated.v1.json"


class PoisonMessage(Exception):  # noqa: N818 - named for the Kafka term
    """The message can never succeed. It goes to the dead-letter topic."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Prediction:
    prob_home_win: float
    model_version: str


Predict = Callable[[dict[str, Any]], Prediction]


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())
    return schema


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def parse_event(value: bytes | None) -> dict[str, Any]:
    if not value:
        raise PoisonMessage("empty message value")
    try:
        event = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoisonMessage(f"invalid JSON: {exc}") from exc
    error = next(iter(_validator().iter_errors(event)), None)
    if error is not None:
        where = "/".join(str(p) for p in error.absolute_path) or "(root)"
        raise PoisonMessage(f"schema mismatch at {where}: {error.message}")
    assert isinstance(event, dict)
    return event


def _start_date(start_time: str) -> str:
    try:
        return datetime.fromisoformat(start_time.replace("Z", "+00:00")).date().isoformat()
    except ValueError as exc:
        raise PoisonMessage(f"bad startTime: {start_time!r}") from exc


def handle_message(ledger: Ledger, predict: Predict, value: bytes | None) -> str:
    """Apply one message. Returns "applied", "duplicate" or "ignored".

    Raises PoisonMessage for bad input. Any other exception is treated by the
    caller as transient.
    """
    event = parse_event(value)
    event_id: str = event["eventId"]
    game: dict[str, Any] = event["game"]
    kind: str = game["statusKind"]

    if ledger.seen(event_id):
        return "duplicate"

    if kind in ("SCHEDULED", "IN_PROGRESS"):
        start_date = _start_date(game["startTime"])
        if ledger.has_game(game["id"]):
            applied = ledger.apply_noop(event_id)
        else:
            prediction = predict({**game, "startDate": start_date})
            applied = ledger.apply_prediction(
                event_id, game, prediction.prob_home_win, prediction.model_version
            )
        return "applied" if applied else "duplicate"

    if kind == "FINAL":
        home, away = game["score"]["home"], game["score"]["away"]
        if home == away:
            raise PoisonMessage(f"final score is a tie: {home}-{away}")
        outcome = "home" if home > away else "away"
        return "applied" if ledger.apply_final(event_id, game, home, away, outcome) else "duplicate"

    # POSTPONED, CANCELED: nothing to record. Remember the id and move on.
    return "ignored" if ledger.apply_noop(event_id) else "duplicate"
