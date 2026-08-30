"""Render `AnalyzeReport` / `CompareReport` to a self-contained HTML report.

Uses a Jinja2 `FileSystemLoader` over the sibling ``templates/`` directory.
The AI narrative is rendered through the ``md`` filter (our dependency-free
``md_to_html``). Templates carry ALL CSS inline, so the returned string is a
fully self-contained document that can be written to disk or served as-is.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ranklens.factors_registry import BY_ID, PHASE_ORDER
from ranklens.models import AnalyzeReport, CompareReport

from .markdown import md_to_html

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["md"] = md_to_html
    env.filters["fmt"] = _fmt_num
    env.filters["pct_strength"] = _pct_strength
    return env


def _fmt_num(v: Any) -> str:
    """Human-friendly number: ints without decimals, floats to 2 dp."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "—"
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f)):,}"
    return f"{f:,.2f}"


def _pct_strength(corr: Any) -> int:
    """|correlation| (0..1) -> integer percent 0..100 for a CSS bar width."""
    try:
        return max(0, min(100, int(round(abs(float(corr)) * 100))))
    except (TypeError, ValueError):
        return 0


def _group_recs_by_phase(recommendations: list) -> list[tuple[str, list]]:
    """Group recommendations into ``(phase, [recs])`` in PHASE_ORDER.

    Only phases that actually have recommendations are returned; recs within a
    phase are sorted by ``priority_score`` descending.
    """
    buckets: dict[str, list] = {}
    for rec in recommendations:
        buckets.setdefault(rec.phase, []).append(rec)

    ordered: list[tuple[str, list]] = []
    seen: set[str] = set()
    for phase in PHASE_ORDER:
        if phase in buckets:
            recs = sorted(buckets[phase], key=lambda r: r.priority_score, reverse=True)
            ordered.append((phase, recs))
            seen.add(phase)
    # Any phase not in PHASE_ORDER (defensive) appended at the end.
    for phase, recs in buckets.items():
        if phase not in seen:
            ordered.append((phase, sorted(recs, key=lambda r: r.priority_score, reverse=True)))
    return ordered


def _sorted_correlations(correlations: list) -> list:
    """All correlations sorted by |best_of_both| descending (None last)."""
    return sorted(
        correlations,
        key=lambda c: abs(c.best_of_both) if c.best_of_both is not None else -1.0,
        reverse=True,
    )


_SPAM_TLDS = (".store", ".top", ".app")


def _host(url: str) -> str:
    """www-stripped lowercase host of a URL (empty on failure)."""
    try:
        net = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001 — best-effort host parse
        return ""
    return net[4:] if net.startswith("www.") else net


def _is_spam_source(bl) -> bool:
    """True for links we hide: junk TLDs (.store/.top/.app) or AS 0–3."""
    dom = (getattr(bl, "source_domain", "") or "").lower()
    if dom.endswith(_SPAM_TLDS):
        return True
    sa = getattr(bl, "source_authority", None)
    return sa is not None and sa <= 3


def _link_strength(bl) -> tuple:
    """Rank a referring domain's links so we can keep its single strongest one."""
    return (
        bl.source_authority or 0.0,
        1 if bl.dofollow else 0,
        bl.topical_relevance or 0.0,
    )


def _backlink_pages(offpage, target_domain: str) -> list[dict]:
    """Group inbound links by destination ranking page — a referring-domains view.

    For each destination ``to_domain`` we keep the single strongest link per
    referring (source) domain, drop spammy/low-authority sources, and sort the
    survivors by source authority. Pages are ordered with the tracked target
    first, then by SERP rank. Powers the prev/next referring-domains viewer.
    Returns ``[{"to_domain","to_rank","is_target","ref_count","links"}, ...]``.
    """
    links = getattr(offpage, "competitor_backlinks", None) if offpage else None
    if not links:
        links = getattr(offpage, "target_backlinks", None) if offpage else None
    if not links:
        return []

    tgt = (target_domain or "").lower()
    order: list[str] = []
    groups: dict[str, dict] = {}
    for bl in links:
        if _is_spam_source(bl):
            continue
        dom = bl.to_domain or "?"
        g = groups.get(dom)
        if g is None:
            g = {
                "to_domain": dom,
                "to_rank": bl.to_rank,
                "is_target": dom.lower() == tgt,
                "by_source": {},
            }
            groups[dom] = g
            order.append(dom)
        src = (bl.source_domain or _host(bl.source_url) or bl.source_url or "").lower()
        prev = g["by_source"].get(src)
        if prev is None or _link_strength(bl) > _link_strength(prev):
            g["by_source"][src] = bl

    pages: list[dict] = []
    for dom in order:
        g = groups[dom]
        rows = sorted(
            g["by_source"].values(),
            key=lambda b: (-(b.source_authority or 0.0), 0 if b.dofollow else 1),
        )
        if not rows:
            continue
        pages.append({
            "to_domain": g["to_domain"],
            "to_rank": g["to_rank"],
            "is_target": g["is_target"],
            "ref_count": len(rows),
            "links": rows,
        })

    pages.sort(key=lambda p: (
        0 if p["is_target"] else 1,
        p["to_rank"] if p["to_rank"] is not None else 999,
    ))
    return pages


def render_analyze(report: AnalyzeReport) -> str:
    """Render an `AnalyzeReport` to a self-contained HTML string."""
    env = _build_env()
    template = env.get_template("report.html.j2")
    return template.render(
        report=report,
        request=report.request,
        target=report.target,
        serp=report.serp,
        ai_html=md_to_html(report.ai_narrative),
        roadmap=_group_recs_by_phase(report.recommendations),
        correlations=_sorted_correlations(report.correlations),
        offpage=report.offpage,
        backlink_pages=_backlink_pages(
            report.offpage,
            _host(report.target.url) if report.target and report.target.url else "",
        ),
        brand=report.brand,
        entity_table=report.entity_table,
        topical=report.topical,
        has_target=bool(report.target and report.target.url),
        by_id=BY_ID,
        phase_order=PHASE_ORDER,
    )


def render_compare(report: CompareReport) -> str:
    """Render a `CompareReport` to a self-contained HTML string."""
    env = _build_env()
    template = env.get_template("compare.html.j2")
    # All moves sorted: biggest absolute movers first, then by after_rank.
    moves = sorted(
        report.moves,
        key=lambda m: (
            -(abs(m.delta) if m.delta is not None else 999),
            m.after_rank if m.after_rank is not None else 999,
        ),
    )
    return template.render(
        report=report,
        request=report.request,
        ai_html=md_to_html(report.ai_narrative),
        moves=moves,
        winners=report.winners,
        losers=report.losers,
        macro=report.macro,
    )


def save_report(html: str, path) -> None:
    """Write a rendered HTML string to ``path`` as UTF-8 (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
