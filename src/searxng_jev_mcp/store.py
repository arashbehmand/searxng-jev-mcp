"""One sqlite file on disk for the cache, engine health and trace (each stdio session starts cold)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
  namespace TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  expires_at REAL NOT NULL,
  PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS engine_health (
  engine TEXT PRIMARY KEY,
  success REAL NOT NULL,
  latency REAL NOT NULL,
  n INTEGER NOT NULL,
  fail_streak INTEGER NOT NULL,
  open_until REAL NOT NULL,
  last_reason TEXT,
  last_probe REAL NOT NULL,
  updated_at REAL NOT NULL
);
"""


class Store:
    """Thread-safe wrapper over one sqlite connection. Operations are tiny, so calls stay synchronous."""

    def __init__(self, path: Path | str):
        self.path = Path(path) if str(path) != ":memory:" else path
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if str(path) != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def execute(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._db.execute(sql, args)
            rows = cur.fetchall()
            self._db.commit()
            return rows

    def close(self) -> None:
        with self._lock:
            self._db.close()


class Cache:
    """Namespaced TTL cache. Values are JSON."""

    def __init__(self, store: Store, *, enabled: bool = True):
        self.store = store
        self.enabled = enabled

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        rows = self.store.execute(
            "SELECT value, expires_at FROM cache WHERE namespace=? AND key=?", (namespace, key)
        )
        if not rows or rows[0]["expires_at"] < time.time():
            return None
        return json.loads(rows[0]["value"])

    def set(self, namespace: str, key: str, value: Any, ttl: float) -> None:
        if not self.enabled or ttl <= 0:
            return
        self.store.execute(
            "INSERT OR REPLACE INTO cache(namespace, key, value, expires_at) VALUES (?,?,?,?)",
            (namespace, key, json.dumps(value, ensure_ascii=False, default=str), time.time() + ttl),
        )

    def purge_expired(self) -> None:
        self.store.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))


class TraceLog:
    """Append-only JSONL log of searches (scrubbed query, engines, timings; no snippets).

    Rotated at `max_bytes`: the current file becomes `trace.jsonl.1` (one backup is kept)."""

    def __init__(self, path: Path, *, enabled: bool = True, max_bytes: int = 10_000_000):
        self.path = path
        self.enabled = enabled
        self.max_bytes = max_bytes

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                self.path.replace(self.path.with_name(self.path.name + ".1"))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"ts": round(time.time(), 3), **record}, ensure_ascii=False, default=str)
                    + "\n"
                )
        except OSError:
            pass
