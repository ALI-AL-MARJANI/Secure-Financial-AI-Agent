"""Append-only JSONL audit log. Raw user text is never written: only a salted hash and guard decisions."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path


class AuditLog:
    def __init__(self, directory: str | Path, salt: str | None = None):
        self.dir = Path(directory)
        # The salt prevents dictionary attacks on short inputs; set AUDIT_SALT in production.
        self.salt = salt if salt is not None else os.environ.get("AUDIT_SALT", "dev-salt")

    def digest(self, text: str) -> str:
        return hashlib.sha256((self.salt + text).encode()).hexdigest()[:16]

    def write(self, event: str, user_text: str | None = None, **fields) -> dict:
        now = dt.datetime.now(dt.UTC)
        entry = {"ts": now.isoformat(timespec="milliseconds"), "event": event}
        if user_text is not None:
            entry["input_hash"] = self.digest(user_text)
        entry.update(fields)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{now.date().isoformat()}.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry
