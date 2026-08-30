"""Automatic, idempotent migration of the legacy SQLite store into PostgreSQL.

Runs on startup: if a pre-Postgres ranklens.db still sits on the data volume, its
users / sessions / runs are copied across with ON CONFLICT DO NOTHING, so nothing
from the SQLite era is lost and re-running is always safe.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ranklens.config import get_settings
from ranklens.db import _conn, ensure_init


def _sqlite_path() -> Path:
    return get_settings().data_dir / "ranklens.db"


def auto_migrate_from_sqlite() -> dict[str, int]:
    """Copy legacy SQLite rows into PostgreSQL. Idempotent. Returns per-table
    inserted counts (0 when nothing to do / file absent)."""
    path = _sqlite_path()
    counts = {"users": 0, "sessions": 0, "runs": 0}
    if not path.exists():
        return counts
    ensure_init()
    sconn = sqlite3.connect(str(path))
    sconn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in sconn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        with _conn() as pg:
            if "users" in tables:
                for row in sconn.execute("SELECT id, email, password_hash, created_at FROM users"):
                    cur = pg.execute(
                        "INSERT INTO users (id, email, password_hash, created_at) "
                        "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                        (row["id"], row["email"], row["password_hash"], row["created_at"]),
                    )
                    counts["users"] += cur.rowcount
            if "sessions" in tables:
                for row in sconn.execute("SELECT token, user_id, created_at, expires_at FROM sessions"):
                    cur = pg.execute(
                        "INSERT INTO sessions (token, user_id, created_at, expires_at) "
                        "VALUES (%s,%s,%s,%s) ON CONFLICT (token) DO NOTHING",
                        (row["token"], row["user_id"], row["created_at"], row["expires_at"]),
                    )
                    counts["sessions"] += cur.rowcount
            if "runs" in tables:
                run_cols = {r[1] for r in sconn.execute("PRAGMA table_info(runs)").fetchall()}
                has_owner = "owner_id" in run_cols
                for row in sconn.execute("SELECT * FROM runs"):
                    cur = pg.execute(
                        "INSERT INTO runs (id, kind, status, label, request, result, error, created_at, finished_at, owner_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                        (
                            row["id"], row["kind"], row["status"], row["label"],
                            row["request"], row["result"], row["error"],
                            row["created_at"], row["finished_at"],
                            (row["owner_id"] if has_owner else None),
                        ),
                    )
                    counts["runs"] += cur.rowcount
    finally:
        sconn.close()
    return counts
