"""Needs a running broker. Set KAFKA_TEST_BROKERS=localhost:29092 to run.

Each run makes its own topics and group, then deletes the topics. It does not
touch `game.updated`, `game.updated.dlq`, or the `predictor` group.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid

import pytest
from events_helpers import FakePredict, make_event, to_bytes

from src.events.consumer import Worker, consumer_config, producer_config
from src.events.ledger import Ledger

BROKERS = os.getenv("KAFKA_TEST_BROKERS")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not BROKERS, reason="KAFKA_TEST_BROKERS not set"),
]


@pytest.fixture
def topics():
    from confluent_kafka.admin import AdminClient, NewTopic

    run = uuid.uuid4().hex[:10]
    main, dlq = f"it.game.updated.{run}", f"it.game.updated.{run}.dlq"
    admin = AdminClient({"bootstrap.servers": BROKERS})
    futures = admin.create_topics([
        NewTopic(main, num_partitions=3, replication_factor=1),
        NewTopic(dlq, num_partitions=1, replication_factor=1),
    ])
    for f in futures.values():
        f.result(timeout=30)
    yield main, dlq, f"it-predictor-{run}"
    for f in admin.delete_topics([main, dlq]).values():
        with contextlib.suppress(Exception):
            f.result(timeout=30)


def run_worker(tmp_path, topic, dlq, group, ledger, predict, until, timeout=60):
    from confluent_kafka import Consumer, Producer

    consumer = Consumer(consumer_config(BROKERS, group_id=group))
    worker = Worker(
        consumer,
        Producer(producer_config(BROKERS)),
        ledger,
        predict,
        topic=topic,
        dlq_topic=dlq,
        group_id=group,
    )
    stop = threading.Event()
    t = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    t.start()
    deadline = time.time() + timeout
    try:
        while time.time() < deadline and not until():
            time.sleep(0.3)
        assert until(), "condition not reached before timeout"
    finally:
        stop.set()
        t.join(timeout=20)
    assert not t.is_alive()


def produce(topic, items):
    from confluent_kafka import Producer

    p = Producer({"bootstrap.servers": BROKERS})
    for key, value in items:
        p.produce(topic, key=key, value=value)
    assert p.flush(30) == 0


def read_all(topic, timeout=30):
    from confluent_kafka import Consumer

    c = Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": f"it-reader-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    c.subscribe([topic])
    out, deadline = [], time.time() + timeout
    while time.time() < deadline and not out:
        m = c.poll(1.0)
        if m is not None and m.error() is None:
            out.append(m)
    c.close()
    return out


def test_duplicate_event_applied_once(tmp_path, topics):
    topic, dlq, group = topics
    gid = f"it-{uuid.uuid4().hex[:8]}"
    ev = to_bytes(make_event(game_id=gid, event_id=str(uuid.uuid4())))
    later = to_bytes(make_event(game_id=gid, status_kind="IN_PROGRESS"))
    # Same key means same partition, so the order is fixed: ev, ev again, later.
    produce(topic, [(gid.encode(), ev), (gid.encode(), ev), (gid.encode(), later)])
    ledger = Ledger(tmp_path / "l.sqlite")
    predict = FakePredict()
    run_worker(
        tmp_path,
        topic,
        dlq,
        group,
        ledger,
        predict,
        until=lambda: ledger.count_events() >= 2,
    )
    time.sleep(2)  # give a wrongly applied duplicate time to show up
    assert ledger.count_events() == 2
    assert [c["id"] for c in predict.calls] == [gid]


def test_bad_message_lands_in_dlq(tmp_path, topics):
    topic, dlq, group = topics
    bad_key = f"bad-{uuid.uuid4().hex[:8]}".encode()
    produce(topic, [(bad_key, b"{not json")])
    ledger = Ledger(tmp_path / "l.sqlite")
    run_worker(
        tmp_path,
        topic,
        dlq,
        group,
        ledger,
        FakePredict(),
        until=lambda: bool(read_all(dlq, timeout=2)),
    )
    msgs = read_all(dlq)
    assert len(msgs) == 1
    m = msgs[0]
    assert m.key() == bad_key and m.value() == b"{not json"
    h = {k: v.decode() for k, v in m.headers()}
    assert h["dlq-source-topic"] == topic
    assert h["dlq-consumer-group"] == group
    assert h["dlq-reason"]
    assert int(h["dlq-source-partition"]) in (0, 1, 2)
    assert int(h["dlq-source-offset"]) >= 0
    assert ledger.count_events() == 0


def test_restart_resumes_from_committed_offset(tmp_path, topics):
    topic, dlq, group = topics
    ids = [f"it-{uuid.uuid4().hex[:8]}" for _ in range(3)]
    produce(topic, [(i.encode(), to_bytes(make_event(game_id=i))) for i in ids[:2]])
    first = Ledger(tmp_path / "first.sqlite")
    run_worker(
        tmp_path,
        topic,
        dlq,
        group,
        first,
        FakePredict(),
        until=lambda: all(first.get_game(i) for i in ids[:2]),
    )
    # New message arrives while the consumer is stopped.
    produce(topic, [(ids[2].encode(), to_bytes(make_event(game_id=ids[2])))])
    # Fresh ledger: anything it holds was read again from Kafka in run two.
    second = Ledger(tmp_path / "second.sqlite")
    run_worker(
        tmp_path,
        topic,
        dlq,
        group,
        second,
        FakePredict(),
        until=lambda: second.get_game(ids[2]) is not None,
    )
    time.sleep(2)
    assert second.get_game(ids[0]) is None
    assert second.get_game(ids[1]) is None
    assert second.count_events() == 1
