"""LLM-scored content effort and helpfulness using an anchored rubric."""
from __future__ import annotations

import json
import statistics

from ranklens.clients.entities import _first_balanced_object, _strip_fences
from ranklens.clients.llm import chat, llm_unavailable
from ranklens.config import get_settings
from ranklens.models import QualityReport


_BATCH_SIZE = 3
_BODY_CHARS = 4000
_COST_PER_CALL = 0.0012
_SYSTEM_PROMPT = """You evaluate web content for the query given by the user.
Return only JSON: {"pages":[{"rank":N,"effort":0-100,"helpfulness":0-100,"note":"15 words maximum"}]}.
Apply these fixed effort anchors: 10 = thin template text; 40 = competent generic coverage; 70 = specific, structured, with some original detail; 90 = original data, first-hand testing, or unique media.
Score helpfulness using Google's helpful-content self-assessment: substantial value versus other results, comprehensive coverage, a satisfying answer, and demonstrated first-hand expertise. Judge only supplied evidence. Use integers and include every supplied rank."""


def _clamp_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:
        return None
    return max(0.0, min(100.0, score))


def _parse_reply(reply: str) -> dict[int, tuple[float, float, str]]:
    if llm_unavailable(reply):
        return {}
    cleaned = _strip_fences(reply)
    candidate = _first_balanced_object(cleaned) or cleaned
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    out: dict[int, tuple[float, float, str]] = {}
    for item in parsed.get("pages", []) if isinstance(parsed, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item.get("rank"))
        except (TypeError, ValueError):
            continue
        effort = _clamp_score(item.get("effort"))
        helpfulness = _clamp_score(item.get("helpfulness"))
        if effort is None or helpfulness is None:
            continue
        note = " ".join(str(item.get("note") or "").split()[:15])
        out[rank] = (effort, helpfulness, note)
    return out


async def _judge_batch(keyword: str, ranks: list[int], bodies_by_rank: dict[int, str], settings):
    pages = "\n\n".join(
        f"RANK {rank}\n{bodies_by_rank[rank][:_BODY_CHARS]}" for rank in ranks
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {keyword}\n\nPages:\n{pages}"},
    ]
    reply = await chat(messages, max_tokens=1000, temperature=0.0, settings=settings)
    return _parse_reply(reply)


async def analyze_quality(
    keyword,
    serp_items,
    bodies_by_rank,
    settings=None,
) -> tuple[QualityReport | None, dict[int, dict[str, float]]]:
    """Judge ranked pages and the tracked target for effort and helpfulness.

    Args:
        keyword: Search query providing the evaluation context.
        serp_items: SERP rows; their ranks determine ranked-page ordering.
        bodies_by_rank: Visible page text keyed by rank, with rank zero as target.
        settings: Application settings including ``ranklens_quality_top_n``.

    Returns:
        The optional quality panel and merge-ready factors keyed by rank. Any
        layer-wide failure returns ``(None, {})`` and never escapes.
    """
    try:
        settings = settings or get_settings()
        bodies = {int(rank): str(body) for rank, body in (bodies_by_rank or {}).items() if str(body).strip()}
        cap = min(10, max(0, int(getattr(settings, "ranklens_quality_top_n", 10))))
        ranked = []
        for item in sorted(serp_items or [], key=lambda row: row.rank):
            if item.rank > 0 and item.rank in bodies and item.rank not in ranked:
                ranked.append(item.rank)
            if len(ranked) >= min(len([rank for rank in bodies if rank > 0]), cap):
                break
        ranks = ranked + ([0] if 0 in bodies else [])
        if not ranks:
            return None, {}

        samples: dict[int, list[tuple[float, float, str]]] = {rank: [] for rank in ranks}
        n_calls = 0
        for start in range(0, len(ranks), _BATCH_SIZE):
            batch = ranks[start:start + _BATCH_SIZE]
            result = await _judge_batch(keyword, batch, bodies, settings)
            n_calls += 1
            for rank, values in result.items():
                if rank in samples:
                    samples[rank].append(values)

        if 0 in samples:
            result = await _judge_batch(keyword, [0], bodies, settings)
            n_calls += 1
            if 0 in result:
                samples[0].append(result[0])

        effort: dict[int, float] = {}
        helpfulness: dict[int, float] = {}
        notes: dict[int, str] = {}
        for rank, values in samples.items():
            if not values:
                continue
            effort[rank] = float(statistics.median(value[0] for value in values))
            helpfulness[rank] = float(statistics.median(value[1] for value in values))
            notes[rank] = values[-1][2]
        if not effort:
            return None, {}

        factors = {
            rank: {"CONTENT_EFFORT": effort[rank], "HELPFULNESS": helpfulness[rank]}
            for rank in effort
        }
        summary = (
            f"Quality rubric scored {len(effort)} page{'s' if len(effort) != 1 else ''} "
            "for content effort and helpfulness."
        )
        report = QualityReport(
            effort=effort,
            helpfulness=helpfulness,
            effort_notes=notes,
            summary=summary,
            cost_usd=_COST_PER_CALL * n_calls,
        )
        return report, factors
    except Exception:
        return None, {}
