"""Kafka consumer for `game.updated`. Run with `python -m src.events.consumer`.

Runs only when KAFKA_BROKERS is set. Offsets are committed by hand, after the
ledger write (or after the dead-letter publish) has succeeded.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

from confluent_kafka import Consumer, Producer

from src.utils.logging import setup as setup_logging

from .handler import PoisonMessage, Predict, handle_message
from .ledger import Ledger, ledger_path
from .predictions import predict_game

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "game.updated"
DEFAULT_DLQ_TOPIC = "game.updated.dlq"
DEFAULT_GROUP = "predictor"


class RetriesExhausted(Exception):  # noqa: N818
    """A transient error lasted through every retry. Nothing was committed."""


class _Stopped(Exception):  # noqa: N818
    """Shutdown was requested while a message was still being retried."""


def consumer_config(brokers: str, group_id: str | None = None) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": group_id or os.getenv("KAFKA_GROUP_ID", DEFAULT_GROUP),
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    }


def producer_config(brokers: str) -> dict[str, Any]:
    return {"bootstrap.servers": brokers}


class Worker:
    def __init__(
        self,
        consumer: Any,
        producer: Any,
        ledger: Ledger,
        predict: Predict,
        *,
        topic: str = DEFAULT_TOPIC,
        dlq_topic: str = DEFAULT_DLQ_TOPIC,
        group_id: str = DEFAULT_GROUP,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.ledger = ledger
        self.predict = predict
        self.topic = topic
        self.dlq_topic = dlq_topic
        self.group_id = group_id
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._stop = threading.Event()

    def run(self, stop: threading.Event, poll_timeout: float = 1.0) -> None:
        self._stop = stop
        self.consumer.subscribe([self.topic])
        try:
            while not stop.is_set():
                msg = self.consumer.poll(poll_timeout)
                if msg is None:
                    continue
                if msg.error():
                    logger.warning("kafka error: %s", msg.error())
                    continue
                try:
                    self.process(msg)
                except _Stopped:
                    break
        finally:
            self.consumer.close()

    def process(self, msg: Any) -> None:
        """Handle one message, retrying transient errors, then commit it."""
        attempt = 0
        while True:
            try:
                self._handle_once(msg)
                break
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise RetriesExhausted(
                        f"gave up on {msg.topic()}[{msg.partition()}]@{msg.offset()}: {exc}"
                    ) from exc
                delay = min(self.base_delay * 2**attempt, self.max_delay)
                logger.warning("transient error, retry %d in %.1fs: %s", attempt + 1, delay, exc)
                self._sleep(delay)
                attempt += 1
                if self._stop.is_set():
                    raise _Stopped from exc
        self.consumer.commit(message=msg, asynchronous=False)

    def _handle_once(self, msg: Any) -> None:
        try:
            result = handle_message(self.ledger, self.predict, msg.value())
        except PoisonMessage as poison:
            logger.warning("poison message, sending to %s: %s", self.dlq_topic, poison.reason)
            self._publish_dlq(msg, poison.reason)
            return
        logger.info("event %s at %s[%d]@%d", result, msg.topic(), msg.partition(), msg.offset())

    def _publish_dlq(self, msg: Any, reason: str) -> None:
        errors: list[Any] = []
        headers = [
            ("dlq-reason", reason),
            ("dlq-source-topic", msg.topic()),
            ("dlq-source-partition", str(msg.partition())),
            ("dlq-source-offset", str(msg.offset())),
            ("dlq-consumer-group", self.group_id),
        ]
        self.producer.produce(
            self.dlq_topic,
            key=msg.key(),
            value=msg.value(),
            headers=headers,
            on_delivery=lambda err, _m: errors.append(err) if err else None,
        )
        remaining = self.producer.flush(30)
        if remaining or errors:
            raise RuntimeError(f"dead-letter publish failed: {errors or f'{remaining} unsent'}")


def main() -> int:
    setup_logging()
    brokers = os.getenv("KAFKA_BROKERS")
    if not brokers:
        logger.info("KAFKA_BROKERS is not set; the consumer is disabled")
        return 0
    group = os.getenv("KAFKA_GROUP_ID", DEFAULT_GROUP)
    worker = Worker(
        Consumer(consumer_config(brokers, group)),
        Producer(producer_config(brokers)),
        Ledger(ledger_path()),
        predict_game,
        topic=os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC),
        dlq_topic=os.getenv("KAFKA_DLQ_TOPIC", DEFAULT_DLQ_TOPIC),
        group_id=group,
        max_retries=int(os.getenv("KAFKA_MAX_RETRIES", "5")),
        sleep=time.sleep,
    )
    stop = threading.Event()
    old = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        worker.run(stop)
    except RetriesExhausted:
        logger.exception("stopping: a message kept failing and was not committed")
        return 1
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
