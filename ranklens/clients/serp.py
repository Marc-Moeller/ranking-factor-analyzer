"""Live SERP via the configured SERP provider.

Endpoint: ``POST {SERP_API_URL}/api/v2/search`` with header ``X-API-Key``.
Returns the top-N organic results as a :class:`~ranklens.models.Serp`.
"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

from ranklens.config import Settings, get_settings
from ranklens.models import Serp, SerpItem


def _domain_of(url: str) -> str:
    """Registrable host, lowercased, ``www.`` stripped."""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


async def fetch_serp(
    keyword: str,
    country: str = "us",
    num: int = 20,
    settings: Settings | None = None,
) -> Serp:
    """Fetch a live top-N organic SERP for ``keyword`` from the configured SERP provider.

    Args:
        keyword: the search query.
        country: 2-letter Google ``gl`` code (``us``, ``au``, ``gb`` ...).
        num: number of results to request (top-N).
        settings: optional pre-loaded settings; falls back to ``get_settings()``.

    Returns:
        A :class:`Serp` with ``source="serp-api"`` and deduped items
        (one per domain, best rank kept), sorted by rank.

    Raises:
        RuntimeError: if ``serp_api_key`` is not configured.
    """
    settings = settings or get_settings()
    if not settings.serp_api_key:
        raise RuntimeError(
            "serp_api_key is not set. Configure SERP_API_KEY in the environment/.env."
        )

    url = f"{settings.serp_api_url.rstrip('/')}/api/v2/search"
    headers = {
        "X-API-Key": settings.serp_api_key,
        "Content-Type": "application/json",
    }
    body = {"query": keyword, "gl": country, "num": num}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results") or []

    # Dedup by domain, keeping the best (lowest) rank.
    best_by_domain: dict[str, SerpItem] = {}
    for r in results:
        url_val = r.get("url") or ""
        if not url_val.lower().startswith("http"):
            continue
        domain = _domain_of(url_val)
        if not domain:
            continue

        rank = r.get("organic_rank")
        if rank is None:
            rank = r.get("position")
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
            title=r.get("title") or "",
            snippet=r.get("snippet") or "",
            displayed_url=r.get("displayed_url") or "",
        )
        existing = best_by_domain.get(domain)
        if existing is None or item.rank < existing.rank:
            best_by_domain[domain] = item

    items = sorted(best_by_domain.values(), key=lambda it: it.rank)

    return Serp(
        keyword=keyword,
        country=country,
        source="serp-api",
        items=items,
    )
