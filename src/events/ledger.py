"""SQLite prediction ledger.

One row per game, plus a table of processed event ids. Every write for an
event runs in one transaction with that event's id, so a replayed event does
nothing.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    start_time      TEXT,
    prob_home_win   REAL,
    model_version   TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    outcome         TEXT,
    settled         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS processed_events (
    event_id TEXT PRIMARY KEY
);
"""


def ledger_path() -> Path:
    override = os.getenv("PREDICTION_LEDGER_PATH")
    if override:
        return Path(override)
    return Path(os.getenv("NBA_DATA_DIR", str(config.DATA_DIR))) / "prediction_ledger.sqlite"


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def read_summary(path: Path) -> dict[str, Any]:
    """Counts and accuracy. Opens the file read-only; never creates it."""
    empty: dict[str, Any] = {"games": 0, "settled": 0, "graded": 0, "accuracy": None}
    if not Path(path).exists():
        return empty
    with closing(_connect_ro(path)) as conn:
        games, settled, graded, correct = conn.execute(
            """
            SELECT
              COUNT(*),
              COALESCE(SUM(settled), 0),
              COALESCE(SUM(settled = 1 AND prob_home_win IS NOT NULL), 0),
              COALESCE(SUM(settled = 1 AND prob_home_win IS NOT NULL
                           AND (prob_home_win >= 0.5) = (outcome = 'home')), 0)
            FROM games
            """
        ).fetchone()
    return {
        "games": games,
        "settled": settled,
        "graded": graded,
        "accuracy": (correct / graded) if graded else None,
    }


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short connection per call keeps this safe across threads.
        # The inner `with conn` commits on success and rolls back on error.
        with closing(sqlite3.connect(self.path, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    # ---- reads ----

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        return dict(row) if row else None

    def has_game(self, game_id: str) -> bool:
        return self.get_game(game_id) is not None

    def seen(self, event_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return row is not None

    def count_events(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM processed_events").fetchone()[0])

    # ---- writes: each returns False when the event id was already applied ----

    def apply_noop(self, event_id: str) -> bool:
        return self._apply(event_id, None)

    def apply_prediction(
        self, event_id: str, game: dict[str, Any], prob: float, model_version: str
    ) -> bool:
        return self._apply(
            event_id, lambda conn: self._upsert_prediction(conn, game, prob, model_version)
        )

    def apply_final(
        self, event_id: str, game: dict[str, Any], home_score: int, away_score: int, outcome: str
    ) -> bool:
        return self._apply(
            event_id,
            lambda conn: self._upsert_final(conn, game, home_score, away_score, outcome),
        )

    def _apply(self, event_id: str, write: Callable[[sqlite3.Connection], None] | None) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("INSERT INTO processed_events (event_id) VALUES (?)", (event_id,))
                if write is not None:
                    write(conn)
        except sqlite3.IntegrityError as exc:
            # Only the event-id insert can collide: the game writes use ON CONFLICT.
            if "processed_events" in str(exc):
                return False
            raise
        return True

    @staticmethod
    def _upsert_prediction(
        conn: sqlite3.Connection, game: dict[str, Any], prob: float, model_version: str
    ) -> None:
        # An existing row, with or without a prediction, is left as it is.
        conn.execute(
            """
            INSERT INTO games (game_id, home_team, away_team, start_time,
                               prob_home_win, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO NOTHING
            """,
            (
                game["id"],
                game["homeTeam"]["name"],
                game["awayTeam"]["name"],
                game["startTime"],
                prob,
                model_version,
            ),
        )

    @staticmethod
    def _upsert_final(
        conn: sqlite3.Connection,
        game: dict[str, Any],
        home_score: int,
        away_score: int,
        outcome: str,
    ) -> None:
        # The prediction columns are never in the UPDATE list.
        conn.execute(
            """
            INSERT INTO games (game_id, home_team, away_team, start_time,
                               home_score, away_score, outcome, settled)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(game_id) DO UPDATE SET
                home_score = excluded.home_score,
                away_score = excluded.away_score,
                outcome    = excluded.outcome,
                settled    = 1
            """,
            (
                game["id"],
                game["homeTeam"]["name"],
                game["awayTeam"]["name"],
                game["startTime"],
                home_score,
                away_score,
                outcome,
            ),
        )
