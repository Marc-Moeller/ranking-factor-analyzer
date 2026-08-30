"""AI report text via the configured OpenAI-compatible LLM endpoint.

``LLM_API_URL`` already ends in ``/v1``, so we hit ``/chat/completions``. Some
upstreams emit a ``<think>...</think>`` reasoning prefix — we strip it before
returning.

If the call fails, we degrade to a short ``[AI report unavailable: ...]``
string so a report can still render without the narrative.
"""
from __future__ import annotations

import asyncio
import re

import httpx

from ranklens.config import Settings, get_settings

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Returned when no LLM provider is configured. Callers must skip their layer
# on this (and on "[AI report unavailable: ...]") rather than treating it as content.
LLM_NO_KEY = "[AI unavailable: no LLM key configured]"

_SECRET_SETTING_NAMES = (
    "llm_api_key",
    "dataforseo_login",
    "dataforseo_password",
    "crux_api_key",
    "serp_api_key",
    "zyte_api_key",
    "embeddings_api_key",
    "authority_api_key",
    "ranklens_api_key",
    "postgres_password",
)


def llm_unavailable(reply: str | None) -> bool:
    """True when ``chat()`` returned a degradation sentinel, not model text."""
    if not reply:
        return True
    text = reply.lstrip()
    return text.startswith("[AI unavailable") or text.startswith("[AI report unavailable")


def _redact(text: str, settings: Settings) -> str:
    """Strip known credential values out of an error string."""
    out = text
    for name in _SECRET_SETTING_NAMES:
        secret = getattr(settings, name, "") or ""
        if len(secret) >= 4 and secret in out:
            out = out.replace(secret, "[redacted]")
    return out


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "", count=1).strip()


async def _post_completion(
    url: str, api_key: str, model: str, messages: list[dict],
    max_tokens: int, temperature: float,
) -> str:
    """One OpenAI-style chat call. Returns the (think-stripped) content or raises."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    return _strip_think(data["choices"][0]["message"]["content"])


async def chat(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 2200,
    temperature: float = 0.4,
    settings: Settings | None = None,
) -> str:
    """Run a chat completion against the configured OpenAI-compatible endpoint.

    Args:
        messages: OpenAI ``[{"role","content"}, ...]`` message list.
        model: model id; defaults to ``llm_model``.
        max_tokens: completion cap.
        temperature: sampling temperature.
        settings: optional pre-loaded settings.

    Returns:
        The assistant text with any leading ``<think>...</think>`` block stripped.
        If no provider is configured, ``LLM_NO_KEY``. If the call fails, a
        string starting with ``"[AI report unavailable: ...]"``.
        Never raises for a missing key.
    """
    settings = settings or get_settings()

    if not settings.llm_api_key:
        return LLM_NO_KEY

    # Rate limits (429) are the dominant transient failure, especially when
    # analyze layers call concurrently — back off and retry before degrading.
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            return await _post_completion(
                f"{settings.llm_api_url.rstrip('/')}/chat/completions",
                settings.llm_api_key,
                model or settings.llm_model,
                messages, max_tokens, temperature,
            )
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            last_err = Exception(_redact(str(e), settings))
            if "429" in str(e) and attempt < 2:
                await asyncio.sleep(4.0 * (attempt + 1))
                continue
            break

    return f"[AI report unavailable: {_redact(str(last_err), settings)}]"
