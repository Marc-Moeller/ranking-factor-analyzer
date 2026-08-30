"""External service clients for the RankLens engine.

Thin async wrappers over the five capabilities Cora needs:
- :func:`fetch_serp` — live top-N SERP (the configured SERP provider).
- :func:`historical_serp` / :func:`live_serp_advanced` — dated + live SERPs (DataForSEO).
- :func:`fetch_pages` — raw HTML fetch (curl_cffi).
- :func:`chat` — AI report text (the configured OpenAI-compatible LLM endpoint).
- :func:`domain_authority` — optional authority/traffic (the configured authority API).

All network functions are async and read credentials via
``ranklens.config.get_settings()``.
"""
from __future__ import annotations

from ranklens.clients.authority import domain_authority
from ranklens.clients.dataforseo import (
    COUNTRY_LOCATION,
    historical_serp,
    live_serp_advanced,
)
from ranklens.clients.fetch import FetchedPage, fetch_pages
from ranklens.clients.llm import chat
from ranklens.clients.serp import fetch_serp

__all__ = [
    "fetch_serp",
    "historical_serp",
    "live_serp_advanced",
    "COUNTRY_LOCATION",
    "fetch_pages",
    "FetchedPage",
    "chat",
    "domain_authority",
]
