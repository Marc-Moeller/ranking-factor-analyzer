"""Deterministic ranking-funnel scoring and competitor explanations."""
from __future__ import annotations

import math
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

from ranklens.factors_registry import BY_ID
from ranklens.models import (
    AnalyzeReport,
    BriefItem,
    CompetitorCard,
    FunnelResult,
    GATE_IDS,
    GateScore,
)


GATE_NAMES = {
    "access": "Access & Experience",
    "lexical": "Lexical Topicality",
    "semantic": "Semantic Relevance",
    "intent": "Intent & Format Fit",
    "entities": "Entity & Topical Completeness",
    "quality": "Quality, Effort & Trust",
    "authority": "Authority",
    "engagement": "Engagement",
}
EXPECTED_CLICK_SHARE = [0.28, 0.15, 0.11, 0.08, 0.07, 0.05, 0.04, 0.04, 0.03, 0.03]
LEXICAL_GROUPS = {group for group in ("Title", "Headings", "Body", "Content", "Keyword")
                  if any(item.group == group for item in BY_ID.values())}


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(value: float, population: list[float]) -> float | None:
    """Return a tie-aware 0-100 percentile against a reference population."""
    values = [item for item in population if math.isfinite(item)]
    if not values:
        return None
    below = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return 100.0 * (below + 0.5 * equal) / len(values)


def _factor_maps(report: AnalyzeReport) -> tuple[dict[int, dict[str, float]], dict[int, object]]:
    factors: dict[int, dict[str, float]] = {}
    pages: dict[int, object] = {}
    for page in report.page_factors or []:
        rank = int(page.rank)
        pages[rank] = page
        factors.setdefault(rank, {}).update({key: value for key, raw in (page.factors or {}).items()
                                             if (value := _finite(raw)) is not None})
    target = factors.setdefault(0, {})
    for correlation in report.correlations or []:
        value = _finite(correlation.target_value)
        if value is not None and correlation.factor_id not in target:
            target[correlation.factor_id] = value
    return factors, pages


def _add_panel_factors(report: AnalyzeReport, factors: dict[int, dict[str, float]]) -> None:
    semantic = report.semantic
    if semantic:
        for rank, value in semantic.best_passage_sim.items():
            factors.setdefault(rank, {}).setdefault("BEST_PASSAGE_SIM", float(value) * 100.0)
        for rank, value in semantic.content_focus.items():
            factors.setdefault(rank, {}).setdefault("CONTENT_FOCUS", float(value) * 100.0)
        from ranklens.analyze.semantic import SUBINTENT_MATCH_THRESHOLD
        threshold = SUBINTENT_MATCH_THRESHOLD.get(semantic.method, 0.45)
        for rank, coverage in semantic.coverage.items():
            if coverage:
                score = 100.0 * sum(float(value) >= threshold for value in coverage.values()) / len(coverage)
                factors.setdefault(rank, {}).setdefault("SUBINTENT_COVERAGE", score)
    quality = report.quality
    if quality:
        for rank, value in quality.effort.items():
            factors.setdefault(rank, {}).setdefault("CONTENT_EFFORT", float(value))
        for rank, value in quality.helpfulness.items():
            factors.setdefault(rank, {}).setdefault("HELPFULNESS", float(value))
    engagement = report.engagement
    if engagement:
        for rank, value in engagement.click_share.items():
            factors.setdefault(rank, {}).setdefault("CLICK_SHARE", float(value) * 100.0)
        for rank, value in engagement.satisfaction.items():
            factors.setdefault(rank, {}).setdefault("SATISFACTION", float(value))


def _normalized_factor_scores(
    factors: dict[int, dict[str, float]], factor_ids: set[str], ranked: list[int]
) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {rank: [] for rank in factors}
    for factor_id in factor_ids:
        population = [factors[rank][factor_id] for rank in ranked if factor_id in factors.get(rank, {})]
        if not population:
            continue
        direction = BY_ID.get(factor_id).direction if factor_id in BY_ID else "more_is_better"
        for rank, page_values in factors.items():
            if factor_id not in page_values:
                continue
            percentile = _percentile(page_values[factor_id], population)
            if percentile is not None:
                out[rank].append(100.0 - percentile if direction == "less_is_better" else percentile)
    return out


def _composite(scores: dict[int, list[float]]) -> dict[int, float]:
    return {rank: mean(values) for rank, values in scores.items() if values}


def _entity_scores(report: AnalyzeReport, factors: dict[int, dict[str, float]], ranked: list[int]) -> dict[int, float]:
    table = report.entity_table
    if not table or table.target_read_ok is False:
        return {}
    ranks = set(ranked) | {0}
    values: dict[int, list[float]] = {rank: [] for rank in ranks}
    normalized = _normalized_factor_scores(factors, {"EAV_COMPLETENESS"}, ranked)
    for rank, scores in normalized.items():
        values.setdefault(rank, []).extend(scores)
    rows = table.coverage_rows or []
    if rows:
        for rank in ranks:
            covered = sum(bool(row.cells.get(rank) and row.cells[rank].present) for row in rows)
            values.setdefault(rank, []).append(covered / len(rows) * 100.0)
    return _composite(values)


def _intent_scores(report: AnalyzeReport, ranked: list[int]) -> dict[int, float]:
    panel = report.intent_fit
    if not panel:
        return {}
    scores: dict[int, float] = {}
    if panel.fit in {"match", "partial", "mismatch"}:
        scores[0] = {"match": 90.0, "partial": 55.0, "mismatch": 15.0}[panel.fit]
    for rank in ranked:
        page_type = panel.page_types.get(rank)
        if page_type and panel.dominant_type:
            scores[rank] = 90.0 if page_type == panel.dominant_type else 40.0
    return scores


def _engagement_scores(report: AnalyzeReport, factors: dict[int, dict[str, float]], ranked: list[int]) -> dict[int, float]:
    scores: dict[int, list[float]] = {rank: [] for rank in set(ranked) | {0}}
    target_rank = report.target.serp_rank if report.target and report.target.serp_rank else None
    for rank in scores:
        click = factors.get(rank, {}).get("CLICK_SHARE")
        if click is not None:
            position = target_rank if rank == 0 else rank
            expected = EXPECTED_CLICK_SHARE[position - 1] if position and 1 <= position <= 10 else 0.02
            scores[rank].append(min(100.0, max(0.0, click / 100.0 / expected * 100.0)))
        satisfaction = factors.get(rank, {}).get("SATISFACTION")
        if satisfaction is not None:
            scores[rank].append(min(100.0, max(0.0, satisfaction)))
    return _composite(scores)


def _weight(per_page: dict[int, float], ranked: list[int]) -> float:
    pairs = [(rank, per_page[rank]) for rank in ranked if rank in per_page]
    if len(pairs) < 3 or len({value for _, value in pairs}) < 2:
        return 0.5
    try:
        with np.errstate(all="ignore"):
            rho = float(spearmanr([value for _, value in pairs], [rank for rank, _ in pairs]).statistic)
        return max(0.25, abs(rho)) if math.isfinite(rho) else 0.5
    except Exception:
        return 0.5


def _details(gate: str, report: AnalyzeReport, factors: dict[int, dict[str, float]]) -> list[str]:
    target = factors.get(0, {})
    if gate == "intent" and report.intent_fit and report.intent_fit.note:
        return [report.intent_fit.note]
    ids = {
        "access": ("CRUX_LCP_MS", "CRUX_INP_MS", "CRUX_CLS"),
        "lexical": ("SALIENT_TERM_COVERAGE", "TITLE_VARS", "BODY_VARS"),
        "semantic": ("BEST_PASSAGE_SIM", "SUBINTENT_COVERAGE", "CONTENT_FOCUS"),
        "entities": ("EAV_COMPLETENESS",),
        "quality": ("CONTENT_EFFORT", "HELPFULNESS"),
        "authority": ("AUTHORITY_SCORE", "PAGE_AUTHORITY", "REF_DOMAINS"),
        "engagement": ("CLICK_SHARE", "SATISFACTION"),
    }.get(gate, ())
    return [f"{BY_ID[factor_id].name}: {target[factor_id]:.1f}"
            for factor_id in ids if factor_id in target and factor_id in BY_ID][:3]


def build_funnel(report: AnalyzeReport) -> FunnelResult | None:
    """Build the target's staged ranking-funnel verdict.

    Args:
        report: Completed analysis with any subset of optional panels.

    Returns:
        A deterministic funnel result, or ``None`` if the input is unusable.
    """
    try:
        factors, pages = _factor_maps(report)
        _add_panel_factors(report, factors)
        ranked = sorted(rank for rank in factors if rank > 0)
        if not ranked:
            ranked = sorted(int(page.rank) for page in report.page_factors if page.rank > 0)

        access_ids = {factor_id for factor_id, meta in BY_ID.items()
                      if meta.group in {"Experience", "Technical"}}
        lexical_ids = {factor_id for factor_id, meta in BY_ID.items() if meta.group in LEXICAL_GROUPS}
        lexical_ids.add("SALIENT_TERM_COVERAGE")
        gate_pages: dict[str, dict[int, float]] = {
            "access": _composite(_normalized_factor_scores(factors, access_ids, ranked)),
            "lexical": _composite(_normalized_factor_scores(factors, lexical_ids, ranked)),
            "semantic": _composite(_normalized_factor_scores(
                factors, {"BEST_PASSAGE_SIM", "SUBINTENT_COVERAGE", "CONTENT_FOCUS"}, ranked)),
            "intent": _intent_scores(report, ranked),
            "entities": _entity_scores(report, factors, ranked),
            "quality": _composite(_normalized_factor_scores(
                factors, {"CONTENT_EFFORT", "HELPFULNESS"} |
                {factor_id for factor_id in BY_ID if factor_id.startswith("TRUST_")}, ranked)),
            "authority": _composite(_normalized_factor_scores(
                factors, {factor_id for factor_id, meta in BY_ID.items() if meta.group == "Authority"}, ranked)),
            "engagement": _engagement_scores(report, factors, ranked),
        }
        access = gate_pages["access"]
        for rank, page in pages.items():
            fetched = 100.0 if getattr(page, "fetched_ok", False) else 0.0
            access[rank] = mean([access[rank], fetched]) if rank in access else fetched
        if 0 not in pages and factors.get(0):
            access[0] = mean([access[0], 100.0]) if 0 in access else 100.0

        gates: list[GateScore] = []
        for gate_id in GATE_IDS:
            per_page = gate_pages.get(gate_id, {})
            target_raw = per_page.get(0)
            population = [per_page[rank] for rank in ranked if rank in per_page]
            score = _percentile(target_raw, population) if target_raw is not None else None
            if gate_id == "entities" and report.entity_table and report.entity_table.target_read_ok is False:
                score = None
                per_page = {}
            verdict = "n/a" if score is None else "pass" if score >= 55 else "weak" if score >= 30 else "fail"
            tier = ("measured" if gate_id == "access" and any(
                factor_id.startswith("CRUX_") for values in factors.values() for factor_id in values)
                    else "estimated" if gate_id in {"intent", "quality", "engagement"}
                    else "computed")
            gates.append(GateScore(gate=gate_id, name=GATE_NAMES[gate_id], score=score,
                                   verdict=verdict, evidence_tier=tier,
                                   weight=_weight(per_page, ranked), details=_details(gate_id, report, factors),
                                   per_page={rank: round(value, 2) for rank, value in per_page.items()}))

        evaluable = [gate for gate in gates if gate.score is not None]
        if evaluable:
            weighted = sum(gate.score * gate.weight for gate in evaluable) / sum(gate.weight for gate in evaluable)
            weak = [gate for gate in evaluable if gate.verdict in {"fail", "weak"}]
            bottleneck = min((gate.score for gate in weak), default=None)
            # Floor the penalty so one zero gate dampens the score instead of
            # erasing it — an all-zero overall reads as a bug, not a diagnosis.
            overall = (weighted * math.sqrt(max(bottleneck, 5.0) / 100.0)
                       if bottleneck is not None else weighted)
        else:
            overall = None
        # The bottleneck is the earliest hard failure in funnel order; only
        # when nothing fails does the earliest weak gate take the label.
        bottleneck_gate = next((gate.gate for gate in gates if gate.verdict == "fail"),
                               next((gate.gate for gate in gates if gate.verdict == "weak"), ""))
        summary = (f"Your first ranking bottleneck is {GATE_NAMES[bottleneck_gate]}."
                   if bottleneck_gate else "No evaluable ranking gate is currently weak.")
        return FunnelResult(gates=gates, bottleneck_gate=bottleneck_gate,
                            overall_score=round(overall, 2) if overall is not None else None, summary=summary)
    except Exception:
        return None


def build_competitor_cards(report: AnalyzeReport, max_cards: int = 5) -> list[CompetitorCard]:
    """Explain the largest gate advantages held by competitors above the target.

    Args:
        report: Completed analysis, preferably with a built funnel.
        max_cards: Maximum number of competitor cards to return.

    Returns:
        Evidence-backed cards in SERP order; an empty list on failure.
    """
    try:
        funnel = report.funnel or build_funnel(report)
        if not funnel or max_cards <= 0:
            return []
        target_rank = report.target.serp_rank if report.target and report.target.found_in_serp else None
        pages = sorted((page for page in report.page_factors if page.rank > 0), key=lambda page: page.rank)
        candidates = [page for page in pages if target_rank is None or page.rank < target_rank][:max_cards]
        cards: list[CompetitorCard] = []
        for page in candidates:
            deltas: list[tuple[float, GateScore, float, float]] = []
            for gate in funnel.gates:
                theirs = gate.per_page.get(page.rank)
                yours = gate.per_page.get(0)
                if theirs is not None and yours is not None and theirs != yours:
                    deltas.append((theirs - yours, gate, theirs, yours))
            # Only genuine advantages explain an outranking; a competitor that
            # trails on every measured gate is ranking on signals we can't see.
            advantages = [item for item in deltas if item[0] > 2.0]
            advantages.sort(key=lambda item: (-item[0], GATE_IDS.index(item[1].gate)))
            reasons = [BriefItem(
                text=(f"Their {gate.name.lower()} score is {theirs:.0f} versus your {yours:.0f}, "
                      f"a {delta:.0f}-point advantage."),
                evidence=f"{gate.name} · {gate.evidence_tier} evidence",
            ) for delta, gate, theirs, yours in advantages[:3]]
            if not reasons:
                reasons = [BriefItem(
                    text=("No measured content gate explains this ranking — the edge is "
                          "likely authority, brand recognition, or behavioral history we "
                          "cannot observe directly."),
                    evidence="all measured gates · no advantage found",
                )]
            cards.append(CompetitorCard(rank=page.rank, domain=page.domain, url=page.url, reasons=reasons))
        return cards
    except Exception:
        return []
