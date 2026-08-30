"""Sitemap discovery + expansion — the domain's own page inventory.

Two-step, never-raises recon:

1. **Discover** sitemap URLs. Primary source is ``robots.txt`` (``Sitemap:`` lines
   — the only declaration Google actually trusts). Fallback is a short list of
   common locations (``/sitemap.xml``, ``/sitemap_index.xml``, ...). Crucially we
   validate the *body* looks like XML, because a lot of WordPress/soft-404 setups
   return ``200`` with an HTML page for a guessed sitemap path (observed in
   testing: ``/sitemap-index.xml`` -> 189 KB of HTML).

2. **Expand** each sitemap into page URLs, recursing through ``<sitemapindex>``
   nests (capped depth) and de-duping by URL so a self-referential index can't
   loop. Fetches use ``curl_cffi`` Chrome impersonation (same as the page fetcher)
   run in a thread so the engine stays async.

Returns a flat, sorted, de-duped list of page URLs — the breadth signal the
topical-authority analyzer matches against the keyword-variation set.
"""
from __future__ import annotations

import asyncio
import re

from curl_cffi import requests as cc

_SITEMAP_LINE_RE = re.compile(r"(?im)^\s*sitemap:\s*(\S+)\s*$")
_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_ACCEPT_LANGUAGE = "en-US,en;q=0.9"

# Common sitemap locations tried only when robots.txt declares none.
_GUESS_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/wp-sitemap.xml",
    "/sitemap/sitemap.xml",
    "/sitemaps.xml",
)

MAX_SITEMAP_FETCHES = 60   # hard cap on total sitemap docs fetched (loop/abuse guard)
MAX_DEPTH = 4              # how deep a sitemapindex nest may go


def _norm_domain(domain: str) -> str:
    """Bare registrable host: strip scheme, path, ``www.`` and trailing slashes."""
    d = domain.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = d.split("/")[0]
    if d.startswith("www."):
        d = d[4:]
    return d.strip()


def _get(url: str) -> str:
    """Blocking GET returning body text (empty string on any failure)."""
    try:
        resp = cc.get(
            url,
            impersonate="chrome",
            timeout=15,
            allow_redirects=True,
            headers={"Accept-Language": _ACCEPT_LANGUAGE},
        )
        if resp.status_code and 200 <= resp.status_code < 400:
            return resp.text or ""
    except Exception:  # noqa: BLE001 — recon is best-effort, never fatal
        pass
    return ""


def _looks_xml(body: str) -> bool:
    """True only for genuine sitemap XML — guards against soft-200 HTML pages."""
    head = body[:600].lstrip().lower()
    return head.startswith("<?xml") or "<urlset" in head or "<sitemapindex" in head


def _is_index(body: str) -> bool:
    return "<sitemapindex" in body[:600].lstrip().lower()


async def discover_sitemaps(domain: str) -> list[str]:
    """Find a domain's sitemap URLs via robots.txt, falling back to guesses.

    Args:
        domain: a bare domain, host, or URL — normalized internally.

    Returns:
        Ordered, de-duped list of sitemap URLs that returned valid XML. Empty if
        none were found (the caller then has no sitemap signal, not an error).
    """
    host = _norm_domain(domain)
    if not host:
        return []

    # 1) robots.txt is the authoritative declaration.
    robots = await asyncio.to_thread(_get, f"https://{host}/robots.txt")
    declared = [u.strip() for u in _SITEMAP_LINE_RE.findall(robots or "")]

    found: list[str] = []
    seen: set[str] = set()
    for u in declared:
        if u and u not in seen:
            seen.add(u)
            found.append(u)
    if found:
        return found

    # 2) No declaration — probe common locations, keep only valid-XML hits.
    async def _probe(path: str) -> str | None:
        body = await asyncio.to_thread(_get, f"https://{host}{path}")
        return path if _looks_xml(body) else None

    hits = await asyncio.gather(*[_probe(p) for p in _GUESS_PATHS])
    for path in hits:
        if path:
            full = f"https://{host}{path}"
            if full not in seen:
                seen.add(full)
                found.append(full)
    return found


async def expand_sitemaps(sitemap_urls: list[str]) -> list[str]:
    """Recursively expand sitemap/sitemapindex docs into page URLs.

    Args:
        sitemap_urls: roots from :func:`discover_sitemaps`.

    Returns:
        Sorted, de-duped list of page (``<loc>``) URLs. Sitemap-index entries are
        followed (bounded by :data:`MAX_DEPTH` and :data:`MAX_SITEMAP_FETCHES`);
        already-seen sitemap URLs are skipped so a cyclic index can't loop.
    """
    pages: set[str] = set()
    seen_sitemaps: set[str] = set()
    fetch_budget = [MAX_SITEMAP_FETCHES]  # list = mutable closure counter

    async def _walk(url: str, depth: int) -> None:
        if depth > MAX_DEPTH or fetch_budget[0] <= 0:
            return
        if url in seen_sitemaps:
            return
        seen_sitemaps.add(url)
        fetch_budget[0] -= 1

        body = await asyncio.to_thread(_get, url)
        if not _looks_xml(body):
            return
        locs = _LOC_RE.findall(body)
        if _is_index(body):
            # Children are themselves sitemaps — recurse (bounded fan-out).
            children = [l for l in locs if l not in seen_sitemaps][: fetch_budget[0]]
            await asyncio.gather(*[_walk(child, depth + 1) for child in children])
        else:
            pages.update(locs)

    await asyncio.gather(*[_walk(u, 0) for u in sitemap_urls])
    return sorted(pages)


async def fetch_inventory(domain: str) -> tuple[list[str], list[str]]:
    """One-call recon: ``(sitemap_urls, page_urls)`` for a domain.

    Convenience wrapper combining :func:`discover_sitemaps` and
    :func:`expand_sitemaps`. Never raises; an unreachable site yields ``([], [])``.
    """
    sitemaps = await discover_sitemaps(domain)
    if not sitemaps:
        return [], []
    pages = await expand_sitemaps(sitemaps)
    return sitemaps, pages
