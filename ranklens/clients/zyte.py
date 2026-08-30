"""Zyte extract API fallback returning raw HTML for hard-to-scrape pages.

When ``curl_cffi`` is blocked or returns thin/JS-gated HTML, a single URL is
escalated here. We hit Zyte's extract endpoint in two stages, cheapest first:

* ``httpResponseBody: true`` -> raw response bytes, **base64-encoded**, no JS.
  Cheap and fast. Tried first.
* ``browserHtml: true`` -> DOM-serialized HTML after a real browser renders and
  runs JS, returned as a **plain string**. More expensive — only tried when the
  cheap path is non-200, errored, or returns thin/blocked HTML.

The two modes are mutually exclusive (one per request). Auth is HTTP Basic with
the API key as the username and a blank password (``auth=(key, "")``).

Everything degrades GRACEFULLY: no ``zyte_api_key`` -> immediate ``None`` (no
escalation, no spend). Every network/parse error is swallowed and turns into
``None`` so the caller can keep the original curl result. This function never
raises.
"""
from __future__ import annotations

import base64

import httpx

from ranklens.config import Settings, get_settings

# Realistic browser UA — a default/empty UA can trip anti-bot (403).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Below this length (after stripping) HTML is treated as thin/blocked.
_MIN_HTML_LEN = 500

# Case-insensitive anti-bot / interstitial markers. Presence => blocked.
_BLOCK_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "attention required",
    "enable javascript",
    "captcha-delivery",
    "px-captcha",
)

# httpResponseBody is a fast HTTP fetch; browserHtml spins up a real browser.
_BODY_TIMEOUT = 60.0
_BROWSER_TIMEOUT = 90.0


def _looks_blocked(html: str) -> bool:
    """True if ``html`` is empty, too short, or carries an anti-bot marker."""
    if not html:
        return True
    stripped = html.strip()
    if len(stripped) < _MIN_HTML_LEN:
        return True
    lowered = stripped.lower()
    return any(marker in lowered for marker in _BLOCK_MARKERS)


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }


async def fetch_via_zyte(
    url: str,
    settings: Settings | None = None,
) -> dict | None:
    """Escalate one URL to the Zyte extract API and return real HTML.

    Tries the cheap ``httpResponseBody`` path first, then escalates to the
    JS-rendering ``browserHtml`` path only if the cheap path is non-200,
    errored, or returns thin/blocked HTML.

    Args:
        url: the single URL to fetch.
        settings: optional pre-loaded settings (key + base URL come from here).

    Returns:
        ``{"html": str, "status_code": int|None, "mode":
        "httpResponseBody"|"browserHtml"}`` on success, or ``None`` on any
        failure, missing key, or thin result. Never raises.
    """
    settings = settings or get_settings()
    key = settings.zyte_api_key
    if not key:
        return None

    endpoint = f"{settings.zyte_api_url.rstrip('/')}/v1/extract"
    auth = (key, "")

    # Step 1 — cheap HTTP path. httpResponseBody is base64-encoded bytes.
    try:
        async with httpx.AsyncClient(timeout=_BODY_TIMEOUT) as client:
            resp = await client.post(
                endpoint,
                auth=auth,
                headers=_headers(),
                json={"url": url, "httpResponseBody": True},
            )
        if resp.status_code == 200:
            data = resp.json()
            status_code = data.get("statusCode")
            encoded = data.get("httpResponseBody")
            if encoded:
                html = base64.b64decode(encoded).decode("utf-8", "replace")
                if not _looks_blocked(html):
                    return {
                        "html": html,
                        "status_code": status_code,
                        "mode": "httpResponseBody",
                    }
    except Exception:  # noqa: BLE001 — transport/parse error: fall through to browser
        pass

    # Step 2 — escalate to the JS-rendering browser path. Plain HTML string.
    try:
        async with httpx.AsyncClient(timeout=_BROWSER_TIMEOUT) as client:
            resp = await client.post(
                endpoint,
                auth=auth,
                headers=_headers(),
                json={"url": url, "browserHtml": True},
            )
        if resp.status_code == 200:
            data = resp.json()
            status_code = data.get("statusCode")
            html = data.get("browserHtml") or ""
            if not _looks_blocked(html):
                return {
                    "html": html,
                    "status_code": status_code,
                    "mode": "browserHtml",
                }
    except Exception:  # noqa: BLE001 — transport/parse error: give up gracefully
        pass

    return None
