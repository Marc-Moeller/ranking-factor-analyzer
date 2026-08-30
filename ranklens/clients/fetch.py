"""Raw HTML page fetch for on-page factor extraction.

No internal service returns raw HTML, so we fetch directly with ``curl_cffi``
(Chrome TLS impersonation beats most anti-bot). The sync ``cc.get`` call is run
in a thread pool so the rest of the engine stays async. Fetches are bounded by
an :class:`asyncio.Semaphore` and every error is captured per-URL.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from curl_cffi import requests as cc

from ranklens.clients.zyte import _looks_blocked, fetch_via_zyte
from ranklens.config import Settings, get_settings

_ACCEPT_LANGUAGE = "en-US,en;q=0.9"


@dataclass
class FetchedPage:
    url: str
    final_url: str
    status_code: int | None
    html: str
    load_ms: float | None
    ok: bool
    error: str | None
    via: str = "curl_cffi"


def _fetch_one(url: str) -> FetchedPage:
    """Blocking single-URL fetch. Never raises — errors land in the dataclass."""
    start = time.perf_counter()
    try:
        resp = cc.get(
            url,
            impersonate="chrome",
            timeout=20,
            allow_redirects=True,
            headers={"Accept-Language": _ACCEPT_LANGUAGE},
        )
        load_ms = (time.perf_counter() - start) * 1000.0
        return FetchedPage(
            url=url,
            final_url=str(getattr(resp, "url", url) or url),
            status_code=resp.status_code,
            html=resp.text or "",
            load_ms=load_ms,
            ok=200 <= resp.status_code < 400,
            error=None,
        )
    except Exception as e:  # noqa: BLE001 — capture all transport/anti-bot failures
        load_ms = (time.perf_counter() - start) * 1000.0
        return FetchedPage(
            url=url,
            final_url=url,
            status_code=None,
            html="",
            load_ms=load_ms,
            ok=False,
            error=str(e),
        )


async def fetch_pages(
    urls: list[str],
    concurrency: int = 12,
    settings: Settings | None = None,
) -> dict[str, FetchedPage]:
    """Fetch raw HTML for many URLs concurrently.

    Args:
        urls: pages to fetch.
        concurrency: max simultaneous fetches.
        settings: optional pre-loaded settings (currently unused beyond defaults,
            accepted for a uniform client signature).

    Returns:
        ``{original_url: FetchedPage}``. Failed fetches have ``ok=False`` and an
        ``error`` string; the original URL is always the dict key.
    """
    settings = settings or get_settings()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(url: str) -> tuple[str, FetchedPage]:
        async with sem:
            page = await asyncio.to_thread(_fetch_one, url)

            # Escalate to Zyte only when curl failed or returned thin/blocked
            # HTML, and only if a Zyte key is configured (else no spend, no-op).
            if settings.zyte_api_key and (
                not page.ok or _looks_blocked(page.html)
            ):
                result = await fetch_via_zyte(url, settings)
                if result is not None:
                    page = FetchedPage(
                        url=url,
                        final_url=url,
                        status_code=result.get("status_code") or 200,
                        html=result["html"],
                        load_ms=page.load_ms,
                        ok=True,
                        error=None,
                        via=f"zyte:{result['mode']}",
                    )
        return url, page

    # Dedup while preserving order so we don't fetch the same URL twice.
    seen: list[str] = []
    for u in urls:
        if u not in seen:
            seen.append(u)

    results = await asyncio.gather(*(_bounded(u) for u in seen))
    return {url: page for url, page in results}
