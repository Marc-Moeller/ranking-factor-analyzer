"""Optional OpenAI-compatible embeddings client for semantic analysis."""
from __future__ import annotations

import math

import httpx

from ranklens.config import Settings, get_settings

EMBEDDING_BATCH_SIZE = 128


async def embed_texts(
    texts: list[str], settings: Settings | None = None,
) -> list[list[float]] | None:
    """Embed text in bounded batches through an OpenAI-compatible endpoint.

    Args:
        texts: Strings to embed, retained in their original order.
        settings: Optional pre-loaded runtime settings.

    Returns:
        One vector per input string, or ``None`` when unconfigured or failed.
        Never raises.
    """
    try:
        settings = settings or get_settings()
        if not settings.embeddings_api_url or not settings.embeddings_api_key:
            return None
        if not texts:
            return []

        url = f"{settings.embeddings_api_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {settings.embeddings_api_key}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        dimension: int | None = None
        async with httpx.AsyncClient(timeout=60.0) as client:
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[start : start + EMBEDDING_BATCH_SIZE]
                response = await client.post(
                    url,
                    headers=headers,
                    json={"model": settings.embeddings_model, "input": batch},
                )
                response.raise_for_status()
                rows = response.json().get("data")
                if not isinstance(rows, list) or len(rows) != len(batch):
                    return None
                rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
                for row in rows:
                    raw = row.get("embedding") if isinstance(row, dict) else None
                    if not isinstance(raw, list) or not raw:
                        return None
                    vector = [float(value) for value in raw]
                    if not all(math.isfinite(value) for value in vector):
                        return None
                    if dimension is None:
                        dimension = len(vector)
                    if len(vector) != dimension:
                        return None
                    vectors.append(vector)
        return vectors if len(vectors) == len(texts) else None
    except Exception:  # noqa: BLE001 — embeddings are an optional enrichment
        return None
