"""Optional domain authority + traffic via a pluggable authority API.

No provider is bundled. If ``authority_api_key`` is unset, returns ``{}`` so
the engine can run without authority data. ``www.`` is stripped before sending
(www-prefixed hosts often fail to resolve) and results are re-keyed back. The
response shape is handled defensively (dict-of-domains or list-of-objects).

A user-supplied endpoint must speak this contract:

**``POST {AUTHORITY_API_URL}/v1/batch-check``**

Headers: ``X-API-Key``, ``Content-Type: application/json``.

Request body::

    {"domains": ["example.com", "other.com", ...]}   # up to 200 per call

Response JSON — any of these wrappers is accepted: the object itself, or a
``results`` / ``data`` / ``domains`` key. Inside that, either:

* a dict keyed by domain whose values are row objects, or
* a list of row objects, each with ``domain`` (or ``host`` / ``url``)

Fields the parser reads from each row::

    authority_score, total_backlinks, referring_domains,
    visits, prev_visits,
    traffic.visits, traffic.prev_visits     # nested traffic object, optional
"""
from __future__ import annotations

from typing import Any

import httpx

from ranklens.config import Settings, get_settings

_BATCH = 200


def _strip_www(domain: str) -> str:
    d = (domain or "").lower()
    if d.startswith("www."):
        d = d[4:]
    return d


def _chunks(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _trend_pct(visits: Any, prev_visits: Any) -> float | None:
    try:
        v = float(visits)
        pv = float(prev_visits)
    except (TypeError, ValueError):
        return None
    if pv > 0:
        return (v - pv) / pv * 100.0
    return None


def _normalize_row(row: dict) -> dict:
    """Map one authority-API record into our flat authority dict."""
    traffic = row.get("traffic") if isinstance(row.get("traffic"), dict) else {}
    visits = row.get("visits", traffic.get("visits"))
    prev_visits = row.get("prev_visits", traffic.get("prev_visits"))
    return {
        "authority_score": row.get("authority_score"),
        "total_backlinks": row.get("total_backlinks"),
        "referring_domains": row.get("referring_domains"),
        "visits": visits,
        "prev_visits": prev_visits,
        "traffic_trend_pct": _trend_pct(visits, prev_visits),
    }


def _extract_results(data: Any) -> dict[str, dict]:
    """Pull a ``{domain: row}`` mapping out of varied response shapes."""
    out: dict[str, dict] = {}

    # Common wrappers: {"results": ...} or {"data": ...} or raw.
    payload = data
    if isinstance(data, dict):
        for key in ("results", "data", "domains"):
            if key in data:
                payload = data[key]
                break

    if isinstance(payload, dict):
        for domain, row in payload.items():
            if isinstance(row, dict):
                out[_strip_www(domain)] = _normalize_row(row)
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            domain = row.get("domain") or row.get("host") or row.get("url")
            if domain:
                out[_strip_www(str(domain))] = _normalize_row(row)

    return out


async def domain_authority(
    domains: list[str],
    settings: Settings | None = None,
) -> dict[str, dict]:
    """Fetch authority + traffic for ``domains`` from the configured authority API.

    Args:
        domains: hostnames (``www.`` is stripped automatically).
        settings: optional pre-loaded settings.

    Returns:
        ``{domain: {"authority_score","total_backlinks","referring_domains",
        "visits","prev_visits","traffic_trend_pct"}}``. Empty dict when no API
        key is configured (graceful) or no data comes back.
    """
    settings = settings or get_settings()
    if not settings.authority_api_key:
        return {}

    # Dedup + strip www, preserving order.
    clean: list[str] = []
    for d in domains:
        sd = _strip_www(d)
        if sd and sd not in clean:
            clean.append(sd)
    if not clean:
        return {}

    url = f"{settings.authority_api_url.rstrip('/')}/v1/batch-check"
    headers = {
        "X-API-Key": settings.authority_api_key,
        "Content-Type": "application/json",
    }

    results: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=120.0) as client:
        for batch in _chunks(clean, _BATCH):
            try:
                resp = await client.post(
                    url, headers=headers, json={"domains": batch}
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:  # noqa: BLE001 — authority is optional, never fatal
                continue
            results.update(_extract_results(data))

    return results
