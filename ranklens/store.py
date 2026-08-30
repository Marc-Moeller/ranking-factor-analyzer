"""Run persistence. PostgreSQL via the shared pool in ranklens.db.

A `Run` is the unit of work the API and CLI both persist. The request + result
are stored as JSON text blobs, so rich report payloads need no schema churn.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from ranklens.db import _conn, ensure_init
from ranklens.models import BYOK_REQUEST_FIELDS, Run, blank_byok_fields


def init_db() -> None:
    ensure_init()


def _scrub_run_credentials(run: Run) -> None:
    """Blank BYOK fields on the Run before any JSON hits Postgres.

    Walks ``run.request`` and ``run.result['request']``, collects the values
    first so they can be redacted out of ``run.error``, then sets every field
    to None. Mutates the Run in place — the in-memory object must not keep a
    copy of the secret after save either.
    """
    secrets: list[str] = []

    def _collect(payload: object) -> None:
        if not isinstance(payload, dict):
            return
        for name in BYOK_REQUEST_FIELDS:
            value = payload.get(name)
            if isinstance(value, str) and value:
                secrets.append(value)

    _collect(run.request)
    if isinstance(run.result, dict):
        _collect(run.result.get("request"))

    if run.error and secrets:
        redacted = run.error
        for secret in secrets:
            if secret and secret in redacted:
                redacted = redacted.replace(secret, "[redacted]")
        run.error = redacted

    blank_byok_fields(run.request)
    if isinstance(run.result, dict):
        blank_byok_fields(run.result.get("request"))


def save_run(run: Run) -> None:
    ensure_init()
    _scrub_run_credentials(run)
    with _conn() as c:
        c.execute(
            """
            INSERT INTO runs (id, kind, status, label, request, result, error, created_at, finished_at, owner_id)
            VALUES (%(id)s, %(kind)s, %(status)s, %(label)s, %(request)s, %(result)s, %(error)s, %(created_at)s, %(finished_at)s, %(owner_id)s)
            ON CONFLICT (id) DO UPDATE SET
                status=EXCLUDED.status, label=EXCLUDED.label, request=EXCLUDED.request,
                result=EXCLUDED.result, error=EXCLUDED.error, finished_at=EXCLUDED.finished_at
            """,
            {
                "id": run.id,
                "kind": run.kind.value,
                "status": run.status.value,
                "label": run.label,
                "request": json.dumps(run.request),
                "result": json.dumps(run.result) if run.result is not None else None,
                "error": run.error,
                "created_at": run.created_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "owner_id": run.owner_id,
            },
        )


def _row_to_run(row: dict, *, with_result: bool) -> Run:
    return Run(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        label=row["label"] or "",
        request=json.loads(row["request"]) if row["request"] else {},
        result=(json.loads(row["result"]) if row["result"] else None) if with_result else None,
        error=row["error"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        owner_id=row["owner_id"],
    )


def get_run(run_id: str) -> Optional[Run]:
    ensure_init()
    with _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
    return _row_to_run(row, with_result=True) if row else None


def list_runs(limit: int = 50, owner_id: Optional[str] = None) -> list[Run]:
    """Most-recent runs. Pass ``owner_id`` to scope to one user; omit it (admin /
    API-key view) to list every run regardless of owner."""
    ensure_init()
    with _conn() as c:
        if owner_id is None:
            rows = c.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT %s", (limit,)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM runs WHERE owner_id = %s ORDER BY created_at DESC LIMIT %s",
                (owner_id, limit),
            ).fetchall()
    return [_row_to_run(r, with_result=False) for r in rows]


def mark_interrupted_runs() -> int:
    """On startup, fail any run still marked 'running' — its background task died
    with the previous process, so it would otherwise hang forever. Returns count."""
    ensure_init()
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "UPDATE runs SET status='error', "
            "error=COALESCE(error,'interrupted by server restart'), finished_at=%s "
            "WHERE status='running'",
            (now,),
        )
        return cur.rowcount
