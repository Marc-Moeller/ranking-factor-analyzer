"""DataForSEO clients — historical (dated) and live SERPs for before/after compare.

HTTP Basic auth (``dataforseo_login`` : ``dataforseo_password``). One keyword per
POST (the API errors on multi-task bodies). Both public functions return a tuple
``(Serp, cost_usd)`` so callers can accumulate spend.
"""
from __future__ import annotations

import base64

import httpx

from ranklens.config import Settings, get_settings
from ranklens.models import Serp, SerpItem

BASE_URL = "https://api.dataforseo.com"

# Google location codes per 2-letter country. Default -> US (2840).
COUNTRY_LOCATION: dict[str, int] = {
    "us": 2840,
    "au": 2036,
    "gb": 2826,
    "ca": 2124,
    "de": 2276,
    "fr": 2250,
    "in": 2356,
    "nz": 2554,
}


def _location_code(country: str) -> int:
    return COUNTRY_LOCATION.get((country or "us").lower(), 2840)


def _auth_header(settings: Settings) -> str:
    if not settings.dataforseo_login or not settings.dataforseo_password:
        raise RuntimeError(
            "DataForSEO credentials missing. Set DATAFORSEO_LOGIN and "
            "DATAFORSEO_PASSWORD in the environment/.env."
        )
    raw = f"{settings.dataforseo_login}:{settings.dataforseo_password}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _sum_cost(data: dict) -> float:
    total = 0.0
    for task in data.get("tasks") or []:
        c = task.get("cost")
        if isinstance(c, (int, float)):
            total += float(c)
    return total


def _dedup_organic(
    raw_items: list[dict],
    num: int,
) -> list[SerpItem]:
    """Filter ``type == organic``, dedup by domain (best rank), top-``num``."""
    best_by_domain: dict[str, SerpItem] = {}
    for it in raw_items or []:
        if it.get("type") != "organic":
            continue
        url_val = it.get("url") or ""
        domain = (it.get("domain") or "").lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if not domain:
            continue
        rank = it.get("rank_group")
        if rank is None:
            continue
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            continue

        item = SerpItem(
            rank=rank,
            url=url_val,
            domain=domain,
            title=it.get("title") or "",
        )
        existing = best_by_domain.get(domain)
        if existing is None or item.rank < existing.rank:
            best_by_domain[domain] = item

    items = sorted(best_by_domain.values(), key=lambda x: x.rank)
    return items[:num]


async def historical_serp(
    keyword: str,
    before_date: str,
    country: str = "us",
    language: str = "en",
    num: int = 20,
    settings: Settings | None = None,
) -> tuple[Serp, float]:
    """Fetch a dated historical SERP snapshot as of just before ``before_date``.

    Picks the snapshot with the latest ``datetime[:10] < before_date``; if none
    qualifies, takes the earliest available snapshot.

    Args:
        keyword: the search query.
        before_date: ISO date (``YYYY-MM-DD``) cutoff for the "before" snapshot.
        country: 2-letter country code -> DataForSEO location_code.
        language: 2-letter language code.
        num: top-N organic results to keep.
        settings: optional pre-loaded settings.

    Returns:
        ``(Serp, cost_usd)`` where Serp.source == "dataforseo-historical" and
        ``snapshot_date`` is the chosen snapshot's date.

    Raises:
        RuntimeError: if creds are missing, or if the keyword has no history
            (``items: null``) — caller should variant-probe a broader term.
    """
    settings = settings or get_settings()
    url = f"{BASE_URL}/v3/dataforseo_labs/google/historical_serps/live"
    headers = {
        "Authorization": _auth_header(settings),
        "Content-Type": "application/json",
    }
    body = [
        {
            "keyword": keyword,
            "location_code": _location_code(country),
            "language_code": language,
        }
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    cost = _sum_cost(data)

    try:
        result0 = data["tasks"][0]["result"][0]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"DataForSEO historical_serps returned no result for {keyword!r}."
        )

    snapshots = result0.get("items")
    if not snapshots:
        raise RuntimeError(
            f"No historical SERP coverage for {keyword!r} "
            f"(items is null/empty). Try a broader variant term."
        )

    # Pick the latest snapshot strictly before the cutoff; else earliest.
    dated = [s for s in snapshots if s.get("datetime")]
    dated.sort(key=lambda s: s["datetime"])
    before = [s for s in dated if s["datetime"][:10] < before_date]
    chosen = before[-1] if before else (dated[0] if dated else None)
    if chosen is None:
        raise RuntimeError(
            f"No usable dated snapshot for {keyword!r} before {before_date}."
        )

    snapshot_date = chosen["datetime"][:10]
    items = _dedup_organic(chosen.get("items") or [], num)

    serp = Serp(
        keyword=keyword,
        country=country,
        language=language,
        source="dataforseo-historical",
        snapshot_date=snapshot_date,
        items=items,
    )
    return serp, cost


async def live_serp_advanced(
    keyword: str,
    depth: int = 20,
    country: str = "us",
    language: str = "en",
    settings: Settings | None = None,
) -> tuple[Serp, float]:
    """Fetch a live SERP right now via DataForSEO's advanced organic endpoint.

    Args:
        keyword: the search query.
        depth: caps the result depth (top-N).
        country: 2-letter country code -> DataForSEO location_code.
        language: 2-letter language code.
        settings: optional pre-loaded settings.

    Returns:
        ``(Serp, cost_usd)`` with Serp.source == "dataforseo-live" and
        ``snapshot_date`` set from the result's capture datetime.

    Raises:
        RuntimeError: if creds are missing or the API returns no result.
    """
    settings = settings or get_settings()
    url = f"{BASE_URL}/v3/serp/google/organic/live/advanced"
    headers = {
        "Authorization": _auth_header(settings),
        "Content-Type": "application/json",
    }
    body = [
        {
            "keyword": keyword,
            "location_code": _location_code(country),
            "language_code": language,
            "depth": depth,
        }
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    cost = _sum_cost(data)

    try:
        result0 = data["tasks"][0]["result"][0]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"DataForSEO live/advanced returned no result for {keyword!r}."
        )

    dt = result0.get("datetime") or ""
    snapshot_date = dt[:10] if dt else None
    items = _dedup_organic(result0.get("items") or [], depth)

    serp = Serp(
        keyword=keyword,
        country=country,
        language=language,
        source="dataforseo-live",
        snapshot_date=snapshot_date,
        items=items,
    )
    return serp, cost
