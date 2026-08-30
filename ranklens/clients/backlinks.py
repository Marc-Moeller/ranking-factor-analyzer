"""Pluggable backlink / authority data provider (Semrush-compatible schema).

No provider is bundled. Configure ``settings.backlink_api_endpoints`` with a
JSON list of ``{"base","key"}`` objects in failover order. On a connection
error / timeout / non-200 we fall through to the next endpoint; the first 200
wins. If the list is empty or invalid, every public function degrades
GRACEFULLY (empty result) so the off-page and brand panels skip rather than
fail the run. Network errors never propagate.

A user-supplied endpoint must speak this contract. Every request is
``POST`` with headers ``X-API-Key``, ``Content-Type: application/json``, and a
Chrome ``User-Agent``.

**``POST {base}/v1/bulk-analysis``**

Request body::

    {"targets": [{"target": "<url-or-domain>", "target_type": "url"|"root_domain"}, ...],
     "concurrency": 8}

Response JSON, fields the parser reads::

    {"results": [
        {"target": "<echoed target>",
         "authority_score": <float>,
         "referring_domains": <int>,
         "total_backlinks": <int>,
         "follow": <int>,
         "nofollow": <int>}
    ]}

For ``target_type="url"`` rows, ``total_backlinks`` / ``referring_domains`` /
``authority_score`` are PAGE-level.

**``POST {base}/v1/backlinks``**

Request body::

    {"target": "<url-or-domain>", "target_type": "url"|"root_domain",
     "limit": <int>, "offset": 0, "sort_field": "page_ascore", "sort_type": "desc"}

Response JSON, fields the parser reads::

    {"backlinks_total": <int>,          # PAGE-level count (not total_backlinks)
     "backlinks": [
        {"source_url": <str>, "anchor": <str>, "is_nofollow": <bool>,
         "source_ascore": <float>, "domain_ascore": <float>, "first_seen": <str>}
     ]}

``total_backlinks`` / ``referring_domains`` / ``authority_score`` on this
response are DOMAIN-level even for ``target_type:url`` — the page-level count
is ``backlinks_total``.

**``POST {base}/v1/ranked-keywords``**

Request body (the domain MUST be under ``domain``, not ``target`` — ``target``
returns 422)::

    {"domain": "<registrable-domain>", "database": "<gl e.g. us>", "limit": <int>}

Response JSON, fields the parser reads::

    {"keywords": [
        {"phrase": <str>, "volume": <float>, "position": <int>, "traffic": <float>}
    ]}
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from ranklens.config import Settings, get_settings

# Realistic browser UA — a default/empty UA can trip Cloudflare (403).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Statuses that mean "this proxy can't serve right now — try the next one".
_FAILOVER_STATUSES = frozenset({502, 503, 504})

_BULK_TIMEOUT = 180.0
_BACKLINKS_TIMEOUT = 120.0
_KEYWORDS_TIMEOUT = 90.0


def _load_proxies(settings: Settings) -> list[dict[str, str]]:
    """Parse ``settings.backlink_api_endpoints`` into a list of ``{"base","key"}``.

    Returns an empty list (graceful) when unset, malformed, or missing the
    required fields — callers treat empty as "no endpoints configured".
    """
    raw = (settings.backlink_api_endpoints or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []

    proxies: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        base = str(item.get("base") or "").rstrip("/")
        key = str(item.get("key") or "")
        if base and key:
            proxies.append({"base": base, "key": key})
    return proxies


def _headers(key: str) -> dict[str, str]:
    return {
        "X-API-Key": key,
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }


def _to_float(value: Any) -> float | None:
    """Coerce to float, returning None on any failure or null."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Coerce to int (via float, to tolerate ``"143"`` / ``143.0``)."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _source_domain(source_url: Any) -> str:
    """netloc of ``source_url``, lowercased, ``www.`` stripped. '' on failure."""
    try:
        netloc = urlparse(str(source_url or "")).netloc.lower()
    except (ValueError, TypeError):
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _first_seen_str(value: Any) -> str | None:
    """Keep ``first_seen`` as a string if present (epoch int or ISO), else None."""
    if value is None or value == "":
        return None
    return str(value)


async def _post_with_failover(
    proxies: list[dict[str, str]],
    path: str,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any] | None:
    """POST ``body`` to ``path`` on each proxy until one returns a 200 JSON dict.

    Falls through to the next proxy on connection error, timeout, 5xx, or 503.
    Returns the parsed JSON dict from the first 200, or None if all fail.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        for proxy in proxies:
            url = f"{proxy['base']}{path}"
            try:
                resp = await client.post(
                    url, headers=_headers(proxy["key"]), json=body
                )
            except Exception:  # noqa: BLE001 — transport error: fail over
                continue

            # 5xx / 503 -> next proxy. Other non-200 (401/422/429) -> also skip.
            if resp.status_code != 200:
                continue

            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
            # Unexpected shape — try the next proxy.
    return None


async def bulk_backlink_stats(
    targets: list[tuple[str, str]],
    settings: Settings | None = None,
) -> dict[str, dict]:
    """Fetch bulk backlink overview counters for up to 200 targets in one call.

    Args:
        targets: list of ``(target, target_type)`` pairs. ``target_type`` is one
            of ``"url"`` / ``"root_domain"``.
        settings: optional pre-loaded settings.

    Returns:
        ``{target_string: row}`` keyed by the EXACT target string passed in. Each
        row is ``{"authority_score","referring_domains","total_backlinks",
        "follow","nofollow"}``. For a ``target_type="url"`` row, the
        ``total_backlinks`` / ``referring_domains`` / ``authority_score`` values
        are PAGE-level. Returns ``{}`` when no proxies are configured or all
        proxies fail.
    """
    settings = settings or get_settings()
    proxies = _load_proxies(settings)
    if not proxies or not targets:
        return {}

    # Cap at 200 per the API limit; preserve order.
    capped = targets[:200]
    body = {
        "targets": [
            {"target": t, "target_type": tt} for (t, tt) in capped
        ],
        "concurrency": 8,
    }

    data = await _post_with_failover(
        proxies, "/v1/bulk-analysis", body, _BULK_TIMEOUT
    )
    if data is None:
        return {}

    rows = data.get("results")
    if not isinstance(rows, list):
        return {}

    out: dict[str, dict] = {}
    # Match returned rows to the target strings we sent. Prefer the row's own
    # `target` field; fall back to positional alignment with `capped`.
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = row.get("target")
        if not key:
            key = capped[idx][0] if idx < len(capped) else None
        if not key:
            continue
        out[str(key)] = {
            "authority_score": _to_float(row.get("authority_score")),
            "referring_domains": _to_int(row.get("referring_domains")),
            "total_backlinks": _to_int(row.get("total_backlinks")),
            "follow": _to_int(row.get("follow")),
            "nofollow": _to_int(row.get("nofollow")),
        }

    return out


async def page_backlinks(
    target: str,
    target_type: str = "url",
    limit: int = 50,
    settings: Settings | None = None,
) -> dict:
    """Fetch the backlinks list for one target.

    Args:
        target: domain or full URL.
        target_type: ``"url"`` (page-level) or ``"root_domain"``.
        limit: max backlinks to request.
        settings: optional pre-loaded settings.

    Returns:
        ``{"backlinks": [...], "page_total_backlinks": int|None}``. Each backlink
        is ``{"source_url","source_domain","anchor","dofollow",
        "source_authority","domain_authority","first_seen"}``. The page-level
        count comes from the response's ``backlinks_total`` field (NOT
        ``total_backlinks``, which is domain-level here). Returns
        ``{"backlinks": [], "page_total_backlinks": None}`` on any failure.
    """
    empty: dict = {"backlinks": [], "page_total_backlinks": None}

    settings = settings or get_settings()
    proxies = _load_proxies(settings)
    if not proxies or not target:
        return empty

    body = {
        "target": target,
        "target_type": target_type,
        "limit": limit,
        "offset": 0,
        "sort_field": "page_ascore",
        "sort_type": "desc",
    }

    data = await _post_with_failover(
        proxies, "/v1/backlinks", body, _BACKLINKS_TIMEOUT
    )
    if data is None:
        return empty

    raw_backlinks = data.get("backlinks")
    backlinks: list[dict] = []
    if isinstance(raw_backlinks, list):
        for item in raw_backlinks:
            if not isinstance(item, dict):
                continue
            source_url = item.get("source_url")
            backlinks.append(
                {
                    "source_url": str(source_url) if source_url else "",
                    "source_domain": _source_domain(source_url),
                    "anchor": str(item.get("anchor") or ""),
                    # is_nofollow -> dofollow = not is_nofollow.
                    "dofollow": not bool(item.get("is_nofollow")),
                    "source_authority": _to_float(item.get("source_ascore")),
                    "domain_authority": _to_float(item.get("domain_ascore")),
                    "first_seen": _first_seen_str(item.get("first_seen")),
                }
            )

    # Page-level count lives in `backlinks_total`, NOT `total_backlinks`.
    return {
        "backlinks": backlinks,
        "page_total_backlinks": _to_int(data.get("backlinks_total")),
    }


async def ranked_keywords(
    domain: str,
    country: str,
    limit: int = 100,
    settings: Settings | None = None,
) -> list[dict]:
    """Fetch a domain's organic ranked keywords (phrase + monthly volume).

    Powers the brand-demand layer: ``/v1/keyword-research`` is currently broken
    (returns ``total_available`` but an empty ``keywords[]``), so we read branded
    search volume from the domain's ranked-keyword rows instead.

    Args:
        domain: registrable domain (NOT a full URL).
        country: Google ``gl`` code -> provider database code (e.g. ``au``/``us``).
        limit: max rows to request.
        settings: optional pre-loaded settings.

    Returns:
        A list of ``{"phrase","volume","position","traffic"}`` rows (volume /
        traffic are floats, position an int; missing values become ``None``).
        Returns ``[]`` when no proxies are configured or all proxies fail.

    Gotcha: the proxy wants the domain under the ``domain`` key — sending
    ``target`` returns a 422.
    """
    settings = settings or get_settings()
    proxies = _load_proxies(settings)
    if not proxies or not domain:
        return []

    body = {
        "domain": domain,
        "database": (country or "us").lower(),
        "limit": limit,
    }

    data = await _post_with_failover(
        proxies, "/v1/ranked-keywords", body, _KEYWORDS_TIMEOUT
    )
    if data is None:
        return []

    rows = data.get("keywords")
    if not isinstance(rows, list):
        return []

    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        phrase = str(row.get("phrase") or "").strip()
        if not phrase:
            continue
        out.append(
            {
                "phrase": phrase,
                "volume": _to_float(row.get("volume")),
                "position": _to_int(row.get("position")),
                "traffic": _to_float(row.get("traffic")),
            }
        )

    return out
