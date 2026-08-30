"""SERP-presentation factors — computed from a ``SerpItem`` alone (no page fetch).

Covers every registry factor whose ``source == "serp"``.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

_YEAR_RE = re.compile(r"(?:19|20)\d\d")
_COM_TLDS = {"com", "net", "org"}


def _tld(domain: str) -> str:
    if not domain:
        return ""
    return domain.rsplit(".", 1)[-1].lower() if "." in domain else ""


def serp_factors(item, vset) -> dict[str, float]:
    out: dict[str, float] = {}

    domain = (item.domain or "").strip().lower()
    title = item.title or ""
    snippet = item.snippet or ""
    url = item.url or ""
    displayed = item.displayed_url or ""

    try:
        out["SR_DOMAIN_COM"] = 1.0 if _tld(domain) in _COM_TLDS else 0.0
    except Exception:
        out["SR_DOMAIN_COM"] = 0.0

    try:
        out["SR_DOMAIN_HYPHEN"] = 1.0 if "-" in domain else 0.0
    except Exception:
        out["SR_DOMAIN_HYPHEN"] = 0.0

    try:
        out["SR_DOMAIN_LEN"] = float(len(domain))
    except Exception:
        out["SR_DOMAIN_LEN"] = 0.0

    try:
        out["SR_URL_HAS_YEAR"] = 1.0 if _YEAR_RE.search(url) else 0.0
    except Exception:
        out["SR_URL_HAS_YEAR"] = 0.0

    try:
        out["SR_SUMMARY_LEN"] = float(len(snippet))
    except Exception:
        out["SR_SUMMARY_LEN"] = 0.0

    try:
        out["SR_TITLE_VARS"] = float(vset.count(title))
    except Exception:
        out["SR_TITLE_VARS"] = 0.0

    # URL variations: prefer the displayed URL; fall back to the real URL path.
    try:
        if displayed:
            url_text = displayed
        else:
            parts = urlsplit(url)
            url_text = f"{parts.netloc} {parts.path}".replace("/", " ").replace("-", " ")
        out["SR_URL_VARS"] = float(vset.count(url_text))
    except Exception:
        out["SR_URL_VARS"] = 0.0

    try:
        out["SR_SUMMARY_VARS"] = float(vset.count(snippet))
    except Exception:
        out["SR_SUMMARY_VARS"] = 0.0

    return out
