"""
A tiny on-disk response cache (SQLite, from Python's standard library).

Why cache?
  - OpenAlex is metered: an identical search should not cost twice.
  - Development: re-running the same question is instant.
  - Later (deep research): many subagents ask overlapping questions.

How it works:
  key   = hash of (URL path + query params)   -> same request, same key
  value = the JSON response
  TTL   = entries older than ttl_seconds are ignored (data can go stale)

Auth params (api_key, mailto) are NOT part of the key: they don't change the answer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# Params that identify WHO is asking, not WHAT is asked.
IGNORED_KEY_PARAMS = frozenset({"api_key", "mailto"})


class ResponseCache:
    def __init__(self, path: Path, ttl_seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(path: str, params: dict[str, Any]) -> str:
        """Stable key: param order doesn't matter, auth params are ignored."""
        relevant = {k: v for k, v in params.items() if k not in IGNORED_KEY_PARAMS}
        blob = json.dumps([path, relevant], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        row = self._db.execute(
            "SELECT value, created_at FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None or time.time() - row[1] > self._ttl:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:  # anything JSON-serializable
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self._db.commit()

    def clear(self) -> None:
        self._db.execute("DELETE FROM responses")
        self._db.commit()

    def close(self) -> None:
        self._db.close()
