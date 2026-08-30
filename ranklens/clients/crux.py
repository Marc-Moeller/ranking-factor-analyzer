"""Chrome UX Report field metrics for ranking-page origins.

CrUX is optional: an absent key or any request failure degrades to missing-data
markers so the ranking pipeline can continue without an access-gate reading.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ranklens.config import Settings, get_settings

_QUERY_URL = "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
_METRICS = {
    "largest_contentful_paint": ("CRUX_LCP_MS", 1.0),
    "interaction_to_next_paint": ("CRUX_INP_MS", 1.0),
    "cumulative_layout_shift": ("CRUX_CLS", 100.0),
}


def _p75(payload: dict[str, Any], metric: str) -> float | None:
    """Extract one finite numeric p75 from a CrUX response."""
    try:
        value = float(payload["record"]["metrics"][metric]["percentiles"]["p75"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


async def crux_metrics(
    origins: list[str],
    settings: Settings | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch phone field-performance p75s for a collection of origins.

    Args:
        origins: Absolute origins such as ``https://example.com``.
        settings: Optional pre-loaded runtime settings.

    Returns:
        A mapping from each unique non-empty origin to its CrUX factors. Origins
        with no record or a failed request receive ``CRUX_HAS_DATA == 0.0``.
        An unconfigured client returns an empty mapping immediately.
    """
    try:
        settings = settings or get_settings()
        if not settings.crux_api_key:
            return {}

        unique = list(dict.fromkeys(origin for origin in origins if origin))
        if not unique:
            return {}

        semaphore = asyncio.Semaphore(8)
        url = f"{_QUERY_URL}?key={settings.crux_api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            async def fetch(origin: str) -> tuple[str, dict[str, float]]:
                async with semaphore:
                    try:
                        response = await client.post(
                            url,
                            json={"origin": origin, "formFactor": "PHONE"},
                        )
                        if response.status_code == 404:
                            return origin, {"CRUX_HAS_DATA": 0.0}
                        response.raise_for_status()
                        payload = response.json()
                        factors: dict[str, float] = {}
                        for metric, (factor_id, multiplier) in _METRICS.items():
                            value = _p75(payload, metric)
                            if value is not None:
                                factors[factor_id] = value * multiplier
                        if not factors:
                            return origin, {"CRUX_HAS_DATA": 0.0}
                        factors["CRUX_HAS_DATA"] = 1.0
                        return origin, factors
                    except Exception:  # noqa: BLE001 — CrUX is optional
                        return origin, {"CRUX_HAS_DATA": 0.0}

            pairs = await asyncio.gather(*(fetch(origin) for origin in unique))
        return dict(pairs)
    except Exception:  # noqa: BLE001 — never fail the analyze pipeline
        return {}
