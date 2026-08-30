"""PostgreSQL connection pool + schema. The single owner of the database.

store.py and auth.py both build on `_conn()`; nothing else talks to the DB
directly. Using a psycopg3 ConnectionPool means the app transparently recovers
if PostgreSQL restarts (the pool reconnects) — part of the durability guarantee
that a DB or server bounce never loses data or wedges the app.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ranklens.config import get_settings

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()
_init_done = False
_init_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                s = get_settings()
                dsn = s.resolved_dsn
                if not s.database_url and not s.postgres_password:
                    raise RuntimeError(
                        "PostgreSQL is not configured. Set POSTGRES_PASSWORD (the app "
                        "builds the DSN from POSTGRES_HOST/USER/DB/PASSWORD), or set a "
                        "full DATABASE_URL=postgresql://user:pass@host:5432/dbname."
                    )
                pool = ConnectionPool(
                    dsn,
                    min_size=1,
                    max_size=10,
                    max_idle=300,
                    kwargs={"row_factory": dict_row},
                    # Validate each connection on checkout and recycle dead ones.
                    # This is what lets the app keep working after PostgreSQL
                    # restarts (server reboot / DB redeploy) instead of handing
                    # out a stale socket and erroring.
                    check=ConnectionPool.check_connection,
                    open=False,
                )
                pool.open(wait=False)
                _pool = pool
    return _pool


@contextmanager
def _conn() -> Iterator:
    """Yield a pooled connection. psycopg3 ConnectionPool commits on clean exit
    and rolls back on exception."""
    pool = _get_pool()
    with pool.connection() as conn:
        yield conn


_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id          text PRIMARY KEY,
    kind        text NOT NULL,
    status      text NOT NULL,
    label       text,
    request     text,
    result      text,
    error       text,
    created_at  text,
    finished_at text,
    owner_id    text
)
"""
_USERS_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id            text PRIMARY KEY,
    email         text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at    text NOT NULL
)
"""
_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    token      text PRIMARY KEY,
    user_id    text NOT NULL,
    created_at text NOT NULL,
    expires_at text NOT NULL
)
"""


def init_schema(retries: int = 30, delay: float = 2.0) -> None:
    """Create all tables. Retries so a cold start where PostgreSQL is not yet
    accepting connections self-heals instead of crash-looping."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            with _conn() as c:
                c.execute(_RUNS_DDL)
                c.execute(_USERS_DDL)
                c.execute(_SESSIONS_DDL)
                c.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS owner_id text")
                c.execute("CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs (owner_id)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at DESC)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)")
            return
        except Exception as e:  # pragma: no cover - depends on DB readiness
            last = e
            time.sleep(delay)
    raise RuntimeError(f"could not initialise PostgreSQL schema: {last}")


def ensure_init() -> None:
    """Idempotent, once-per-process schema init. Cheap to call on every DB op."""
    global _init_done
    if _init_done:
        return
    with _init_lock:
        if _init_done:
            return
        init_schema()
        _init_done = True
