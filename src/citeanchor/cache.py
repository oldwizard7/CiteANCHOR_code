"""SQLite cache of retrieved records, keyed by normalized title."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .records import Work, norm_title

DEFAULT_PATH = Path(__file__).with_name("cache.sqlite3")


class RecordCache:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self._local = threading.local()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS works (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                source TEXT,
                hits INTEGER DEFAULT 0
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_source ON works(source)")

    def _conn(self) -> sqlite3.Connection:
        # One connection per thread: sqlite3 connections cannot be shared across threads.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def key_for(title: str) -> str:
        return norm_title(title)

    def get(self, title: str) -> list[Work]:
        key = self.key_for(title)
        if not key:
            return []
        with self._conn() as c:
            row = c.execute("SELECT payload FROM works WHERE key = ?", (key,)).fetchone()
            if not row:
                return []
            c.execute("UPDATE works SET hits = hits + 1 WHERE key = ?", (key,))
        return [Work(**w) for w in json.loads(row[0])]

    def put(self, title: str, works: list[Work]) -> None:
        key = self.key_for(title)
        if not key or not works:
            return
        payload = json.dumps([w.to_dict() for w in works], ensure_ascii=False)
        with self._conn() as c:
            c.execute(
                "INSERT INTO works(key, payload, source) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                (key, payload, works[0].source))

    def stats(self) -> dict:
        with self._conn() as c:
            n, hits = c.execute("SELECT COUNT(*), COALESCE(SUM(hits),0) FROM works").fetchone()
        return {"entries": n, "hits": hits, "path": str(self.path)}
