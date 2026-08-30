"""LLM-classified SERP page formats and target intent fit."""
from __future__ import annotations

import json
from collections import Counter

from ranklens.clients.entities import _first_balanced_object, _strip_fences
from ranklens.clients.llm import chat, llm_unavailable
from ranklens.config import get_settings
from ranklens.models import IntentFit, SerpItem

PAGE_TYPES = {
    "listicle", "guide", "product", "category", "service", "tool",
    "forum", "news", "comparison", "other",
}
MAX_RANKED_PAGES = 10
MAX_BODY_WORDS = 120
MAX_TOKENS = 900
COST_PER_CALL_USD = 0.0012

_SYSTEM_PROMPT = (
    "Classify search-result pages by the format and intent they present for the "
    "given query. Return JSON only, with exactly this shape: "
    '{"pages":[{"rank":1,"page_type":"guide","commercial":false}],'
    '"is_ymyl":false,"serp_features_note":""}. '
    "page_type must be one of: listicle, guide, product, category, service, "
    "tool, forum, news, comparison, other. Preserve every supplied rank and "
    "classify each page once. commercial and is_ymyl must be JSON booleans."
)


def _same_url(left: str, right: str) -> bool:
    return left.strip().rstrip("/").casefold() == right.strip().rstrip("/").casefold()


def _parse_reply(reply: str) -> dict | None:
    cleaned = _strip_fences(reply)
    for candidate in (cleaned, _first_balanced_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _plain_type(page_type: str, count: int = 1) -> str:
    if page_type == "other":
        return "other page" if count == 1 else "other pages"
    return f"{page_type} page" if count == 1 else f"{page_type} pages"


async def analyze_intent(
    keyword: str,
    serp_items: list[SerpItem],
    bodies_by_rank: dict[int, str],
    target_url: str | None,
    settings=None,
) -> IntentFit | None:
    """Classify ranking formats and judge whether the tracked target fits them.

    Args:
        keyword: Query whose result-page intent is being assessed.
        serp_items: Ranked organic results, including title metadata.
        bodies_by_rank: Visible page text keyed by rank; rank 0 is the target.
        target_url: Tracked URL, when the analysis has one.
        settings: Pre-loaded application settings for the LLM client.

    Returns:
        The classified intent panel, or ``None`` when classification fails.
    """
    try:
        settings = settings or get_settings()
        ranked = sorted((item for item in serp_items if item.rank > 0), key=lambda item: item.rank)
        top_ranked = ranked[:MAX_RANKED_PAGES]
        target_item = next(
            (item for item in ranked if target_url and _same_url(item.url, target_url)),
            None,
        )

        requested: list[tuple[int, str, str]] = [
            (item.rank, item.title, bodies_by_rank.get(item.rank, ""))
            for item in top_ranked
        ]
        if target_url and 0 in bodies_by_rank:
            requested.append((0, target_item.title if target_item else "", bodies_by_rank[0]))
        elif target_item and target_item.rank not in {rank for rank, _, _ in requested}:
            requested.append(
                (target_item.rank, target_item.title, bodies_by_rank.get(target_item.rank, ""))
            )

        if not requested or not top_ranked:
            return None

        page_blocks = []
        for rank, title, body in requested:
            excerpt = " ".join((body or "").split()[:MAX_BODY_WORDS])
            page_blocks.append(f"Rank {rank}\nTitle: {title or '(untitled)'}\nBody: {excerpt or '(unavailable)'}")
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {keyword}\n\nPages:\n\n" + "\n\n".join(page_blocks),
            },
        ]
        # Two attempts: the provider stack 429s under the concurrent funnel
        # load, and intent fit is the highest-leverage single call in the run.
        parsed = None
        for _ in range(2):
            reply = await chat(
                messages, max_tokens=MAX_TOKENS, temperature=0.0, settings=settings
            )
            if llm_unavailable(reply):
                continue
            parsed = _parse_reply(reply)
            if parsed and isinstance(parsed.get("pages"), list):
                break
            parsed = None
        if not parsed:
            return None

        allowed_ranks = {rank for rank, _, _ in requested}
        page_types: dict[int, str] = {}
        for page in parsed["pages"]:
            if not isinstance(page, dict):
                continue
            rank = page.get("rank")
            page_type = str(page.get("page_type") or "").strip().casefold()
            if isinstance(rank, bool):
                continue
            try:
                rank = int(rank)
            except (TypeError, ValueError):
                continue
            if rank in allowed_ranks and page_type in PAGE_TYPES:
                page_types[rank] = page_type

        competitor_ranks = [item.rank for item in top_ranked if item.rank in page_types]
        if not competitor_ranks:
            return None
        counts = Counter(page_types[rank] for rank in competitor_ranks)
        dominant_type, dominant_count = counts.most_common(1)[0]
        dominant_share = dominant_count / len(competitor_ranks)

        target_rank = 0 if 0 in page_types else (target_item.rank if target_item else None)
        target_type = page_types.get(target_rank, "") if target_rank is not None else ""
        if target_type == dominant_type and dominant_share >= 0.4:
            fit = "match"
        elif target_type and target_type != dominant_type and dominant_share >= 0.5:
            fit = "mismatch"
        else:
            fit = "partial"

        if target_type:
            verdict = {
                "match": "format match",
                "partial": "mixed format fit",
                "mismatch": "format mismatch",
            }[fit]
            note = (
                f"{dominant_count}/{len(competitor_ranks)} ranking pages are "
                f"{_plain_type(dominant_type, dominant_count)}; your page is "
                f"{_plain_type(target_type)} — {verdict}."
            )
        else:
            note = (
                f"{dominant_count}/{len(competitor_ranks)} ranking pages are "
                f"{_plain_type(dominant_type, dominant_count)}; no target page "
                "type was available — fit is partial."
            )

        features_note = str(parsed.get("serp_features_note") or "").strip()
        return IntentFit(
            page_types=page_types,
            dominant_type=dominant_type,
            dominant_share=dominant_share,
            target_type=target_type,
            fit=fit,
            serp_features=[features_note] if features_note else [],
            is_ymyl=parsed.get("is_ymyl") is True,
            note=note,
            cost_usd=COST_PER_CALL_USD,
        )
    except Exception:  # noqa: BLE001 - optional analysis layers never break a run
        return None
