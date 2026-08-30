"""Estimated SERP clicks and post-click satisfaction via small LLM panels."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from ranklens.clients.entities import _first_balanced_object, _strip_fences
from ranklens.clients.llm import chat, llm_unavailable
from ranklens.config import Settings, get_settings
from ranklens.models import EngagementReport, SerpItem

_COST_PER_CALL = 0.0012
_PANEL_SIZE = 12


def _json_object(text: str) -> dict[str, Any] | None:
    """Parse a fenced or prose-wrapped JSON object defensively."""
    if llm_unavailable(text):
        return None
    clean = _strip_fences(text)
    candidate = _first_balanced_object(clean)
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _target_item(serp_items: list[SerpItem], target_url: str | None) -> SerpItem | None:
    if not target_url:
        return None
    needle = target_url.rstrip("/").lower()
    return next(
        (item for item in serp_items if item.url.rstrip("/").lower() == needle),
        None,
    )


def _click_cards(
    keyword: str,
    serp_items: list[SerpItem],
    target_url: str | None,
    top_n: int,
) -> tuple[list[tuple[str, SerpItem]], dict[str, int]]:
    """Build one deterministically shuffled, neutrally identified result panel."""
    ranked = sorted(serp_items, key=lambda item: item.rank)
    selected = ranked[:max(0, top_n)]
    target = _target_item(ranked, target_url)
    if target and all(item.rank != target.rank for item in selected):
        selected.append(target)

    seed = int.from_bytes(hashlib.sha256(keyword.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(selected)
    cards: list[tuple[str, SerpItem]] = []
    id_to_rank: dict[str, int] = {}
    for index, item in enumerate(selected):
        neutral_id = chr(65 + index) if index < 26 else f"R{index + 1}"
        cards.append((neutral_id, item))
        id_to_rank[neutral_id] = 0 if target is item else item.rank
    return cards, id_to_rank


async def _panel_votes(
    keyword: str,
    cards: list[tuple[str, SerpItem]],
    id_to_rank: dict[str, int],
    settings: Settings,
) -> tuple[dict[int, float], dict[int, tuple[float, str]]] | None:
    result_lines = [
        f"ID {neutral_id}\nTitle: {item.title}\nSnippet: {item.snippet}"
        for neutral_id, item in cards
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You simulate independent search-result choices exactly as instructed. "
                "Use only the neutral result ids; never infer or favor search position."
            ),
        },
        {
            "role": "user",
            "content": (
                f"You are {_PANEL_SIZE} different searchers (mixed: in a hurry, thorough, "
                f"skeptical, price-sensitive, and comparison-minded) searching {keyword!r}. "
                "Here are the results in RANDOM order. Each searcher clicks exactly one. "
                "Return JSON only as {\"votes\": [{\"id\": \"A\", \"count\": 3, "
                "\"why\": \"short reason\"}]}. Counts must total 12.\n\n"
                + "\n\n".join(result_lines)
            ),
        },
    ]
    reply = await chat(messages, max_tokens=700, temperature=0.0, settings=settings)
    payload = _json_object(reply)
    if not payload or not isinstance(payload.get("votes"), list):
        return None

    counts = {rank: 0.0 for rank in id_to_rank.values()}
    reasons: dict[int, tuple[float, str]] = {}
    for vote in payload["votes"]:
        if not isinstance(vote, dict):
            continue
        neutral_id = str(vote.get("id", "")).strip().upper()
        rank = id_to_rank.get(neutral_id)
        try:
            count = max(0.0, float(vote.get("count", 0)))
        except (TypeError, ValueError):
            continue
        if rank is None:
            continue
        counts[rank] += count
        why = str(vote.get("why") or "").strip()
        previous = reasons.get(rank)
        if why and (previous is None or count > previous[0]):
            reasons[rank] = (count, why)
    total = sum(counts.values())
    if total <= 0:
        return None
    return ({rank: count / total for rank, count in counts.items()}, reasons)


async def _satisfaction(
    keyword: str,
    ranks: list[int],
    bodies_by_rank: dict[int, str],
    settings: Settings,
) -> tuple[dict[int, float], dict[int, str]] | None:
    pages = []
    for rank in ranks:
        words = (bodies_by_rank.get(rank) or "").split()[:200]
        if words:
            pages.append(f"RANK {rank}: {' '.join(words)}")
    if not pages:
        return None
    messages = [
        {
            "role": "system",
            "content": "Judge post-click satisfaction from the supplied excerpts. Return JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"For each page, you just clicked it for {keyword!r}. Score 0-100 confidence "
                "you will get what you came for, and name the first friction point in at most "
                "12 words. Return {\"pages\": [{\"rank\": 1, \"satisfaction\": 75, "
                "\"friction\": \"answer is buried\"}]}.\n\n" + "\n\n".join(pages)
            ),
        },
    ]
    reply = await chat(messages, max_tokens=700, temperature=0.0, settings=settings)
    payload = _json_object(reply)
    if not payload or not isinstance(payload.get("pages"), list):
        return None
    scores: dict[int, float] = {}
    friction: dict[int, str] = {}
    allowed = set(ranks)
    for page in payload["pages"]:
        if not isinstance(page, dict):
            continue
        try:
            rank = int(page.get("rank"))
            score = min(100.0, max(0.0, float(page.get("satisfaction"))))
        except (TypeError, ValueError):
            continue
        if rank not in allowed:
            continue
        scores[rank] = score
        friction[rank] = str(page.get("friction") or "").strip()
    return (scores, friction) if scores else None


async def analyze_engagement(
    keyword: str,
    serp_items: list[SerpItem],
    bodies_by_rank: dict[int, str],
    target_url: str | None,
    settings: Settings | None = None,
) -> tuple[EngagementReport | None, dict[int, dict[str, float]]]:
    """Estimate result clicks and early post-click satisfaction.

    Args:
        keyword: Search query shown to the simulated panels.
        serp_items: Organic results carrying titles and snippets.
        bodies_by_rank: Visible page text keyed by rank; rank zero is the target.
        target_url: Optional tracked URL, used to identify its SERP result.
        settings: Optional pre-loaded runtime settings.

    Returns:
        ``(EngagementReport, factors_by_rank)`` when any estimate succeeds, or
        ``(None, {})`` on total failure. The function never raises.
    """
    try:
        settings = settings or get_settings()
        cards, id_to_rank = _click_cards(
            keyword, serp_items, target_url, settings.ranklens_engagement_top_n
        )
        panel_results = []
        calls = 0
        if cards:
            # Two agreeing panels, with headroom for transient provider
            # failures when the funnel layers all hit the LLM concurrently.
            for _ in range(4):
                if len(panel_results) >= 2:
                    break
                calls += 1
                result = await _panel_votes(keyword, cards, id_to_rank, settings)
                if result:
                    panel_results.append(result)

        click_share: dict[int, float] = {}
        click_reasons: dict[int, str] = {}
        if panel_results:
            ranks = set().union(*(shares.keys() for shares, _ in panel_results))
            click_share = {
                rank: sum(shares.get(rank, 0.0) for shares, _ in panel_results) / len(panel_results)
                for rank in ranks
            }
            total = sum(click_share.values())
            if total > 0:
                click_share = {rank: share / total for rank, share in click_share.items()}
            reason_candidates: dict[int, list[tuple[float, str]]] = {}
            for _, reasons in panel_results:
                for rank, reason in reasons.items():
                    reason_candidates.setdefault(rank, []).append(reason)
            click_reasons = {
                rank: max(candidates, key=lambda candidate: candidate[0])[1]
                for rank, candidates in reason_candidates.items()
            }

        satisfaction_ranks = ([0] if bodies_by_rank.get(0) else []) + [
            rank for rank in sorted(r for r in bodies_by_rank if r > 0)[:3]
        ]
        satisfaction: dict[int, float] = {}
        friction: dict[int, str] = {}
        if satisfaction_ranks:
            calls += 1
            sat_result = await _satisfaction(keyword, satisfaction_ranks, bodies_by_rank, settings)
            if sat_result:
                satisfaction, friction = sat_result

        if not click_share and not satisfaction:
            return None, {}

        factors: dict[int, dict[str, float]] = {}
        for rank, share in click_share.items():
            factors.setdefault(rank, {})["CLICK_SHARE"] = min(100.0, max(0.0, share * 100.0))
        for rank, score in satisfaction.items():
            factors.setdefault(rank, {})["SATISFACTION"] = score

        target_share = click_share.get(0)
        if target_share is not None:
            summary = f"Your title wins {target_share * 100:.0f}% of simulated clicks."
        else:
            summary = "Simulated searchers reveal how strongly each result earns the click."
        report = EngagementReport(
            click_share=click_share,
            click_reasons=click_reasons,
            satisfaction=satisfaction,
            friction=friction,
            summary=summary,
            cost_usd=_COST_PER_CALL * calls,
        )
        return report, factors
    except Exception:  # noqa: BLE001 — engagement is an optional analyze layer
        return None, {}
