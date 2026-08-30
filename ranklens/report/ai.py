"""AI narrative generators for RankLens reports.

Two best-effort functions that turn a finished `AnalyzeReport` / `CompareReport`
into a markdown narrative via the LLM router (`ranklens.clients.llm.chat`). Both
build a COMPACT, data-grounded prompt — only the signal the model needs, never a
dump of every factor/page — and degrade to a short markdown fallback on any
failure so a report always renders.
"""
from __future__ import annotations

from ranklens.clients import llm
from ranklens.factors_registry import BY_ID
from ranklens.models import AnalyzeReport, CompareReport

_FALLBACK = "_AI narrative unavailable._"

_ANALYZE_SYSTEM = (
    "You are a senior technical SEO analyst. You reason from on-page correlation "
    "data for a single keyword's SERP. You are precise, cite the actual numbers, "
    "and never overclaim causation from correlation. Output GitHub-flavoured "
    "Markdown only — no preamble, no code fences around the whole answer. "
    "Write all math, symbols and arrows as plain text/Unicode (e.g. 'r = -0.45', "
    "'→'); never use LaTeX or $...$ math — the report has no math renderer."
)

_COMPARE_SYSTEM = (
    "You are a senior SEO analyst specialising in Google core-update forensics. "
    "Your interpretive frame: a core update is Google re-deciding, per query, "
    "which result best satisfies users. In YMYL / high-stakes queries the "
    "site-authority prior tends to dominate; in commodity / commercial verticals "
    "per-query behavioural relevance (clicks, dwell, intent match) tends to "
    "dominate. You reason honestly within this frame, flag when third-party "
    "metrics are soft, and never pretend to know Google's internals. Output "
    "GitHub-flavoured Markdown only. Write all math, symbols and arrows as plain "
    "text/Unicode (e.g. 'r = -0.45', '→'); never use LaTeX or $...$ math — the "
    "report has no math renderer."
)


def _unit(factor_id: str) -> str:
    fd = BY_ID.get(factor_id)
    return fd.unit if fd else "value"


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def _analyze_prompt(report: AnalyzeReport) -> str:
    req = report.request
    has_target = report.target is not None and report.target.url

    # Top significant factors by |best_of_both|, capped at 15.
    sig = [c for c in report.correlations if c.significant and c.best_of_both is not None]
    sig.sort(key=lambda c: abs(c.best_of_both or 0.0), reverse=True)
    sig = sig[:15]

    lines: list[str] = []
    lines.append(f"KEYWORD: {req.keyword!r}")
    lines.append(
        f"SERP: {report.n_pages_analyzed} pages analyzed "
        f"({report.pages_fetched_ok} fetched OK), "
        f"country={req.country}, language={req.language}, "
        f"significance |r| > {report.significance_threshold:.2f}"
    )
    if has_target:
        t = report.target
        lines.append(
            f"TARGET URL: {t.url} — optimization_score={t.optimization_score:.0f}/100, "
            f"{t.factors_met}/{t.factors_total} significant factors met, "
            f"quick_wins={t.quick_wins}, "
            f"serp_rank={t.serp_rank if t.found_in_serp else 'not ranking in top-N'}"
        )
    else:
        lines.append("TARGET URL: none (SERP-only analysis, no page graded)")

    lines.append("")
    lines.append("TOP SIGNIFICANT FACTORS (name | signed correlation | page-1 avg"
                 + (" | your value | deficit-to-goal" if has_target else "") + "):")
    for c in sig:
        row = f"- {c.name} ({c.group}) | r={c.best_of_both:+.2f} | p1_avg={_fmt(c.page1_avg)} {_unit(c.factor_id)}"
        if has_target:
            dv = _fmt(c.target_value)
            df = _fmt(c.deficit)
            row += f" | you={dv} | deficit={df}"
        lines.append(row)

    # Top roadmap actions, capped at 10.
    recs = sorted(report.recommendations, key=lambda r: r.priority_score, reverse=True)[:10]
    if recs:
        lines.append("")
        lines.append("TOP ROADMAP ACTIONS (priority order):")
        for r in recs:
            tag = " [Top-200]" if r.top200 else ""
            lines.append(
                f"- {r.name} ({r.phase}, {r.difficulty}){tag}: {r.action_text} "
                f"[current={_fmt(r.current)} -> goal={_fmt(r.goal)}]"
            )

    lines.append("")
    lines.append(
        "WRITE THE REPORT with these sections (use ## headings):\n"
        "1. **Executive Summary** — 2-3 sentences on what is driving rankings for "
        "THIS keyword, grounded in the strongest factors above.\n"
        "2. **The 5 Highest-Leverage Changes** — a numbered list, priority order, "
        "each citing the concrete current->goal numbers and why it matters.\n"
        "3. **Honest Caveat** — one short paragraph: correlation is not causation, "
        f"this is {report.n_pages_analyzed} pages (small sample), and these are "
        "on-page factors only.\n"
        "Target ~500-700 words. Be specific with numbers. No fluff."
    )
    return "\n".join(lines)


def _compare_prompt(report: CompareReport) -> str:
    req = report.request
    update = req.update_name or "the algorithm update"

    def _winrow(m) -> str:
        bits = [m.domain]
        if m.before_rank is None and m.after_rank is not None:
            bits.append(f"entered at #{m.after_rank}")
        elif m.after_rank is None and m.before_rank is not None:
            bits.append(f"dropped out (was #{m.before_rank})")
        elif m.before_rank is not None and m.after_rank is not None:
            arrow = "up" if (m.delta or 0) > 0 else ("down" if (m.delta or 0) < 0 else "flat")
            bits.append(f"#{m.before_rank} -> #{m.after_rank} ({arrow} {abs(m.delta or 0)})")
        if m.authority_score is not None:
            bits.append(f"authority={m.authority_score:.0f}")
        if m.traffic_trend_pct is not None:
            bits.append(f"traffic_trend={m.traffic_trend_pct:+.0f}%")
        return " | ".join(bits)

    lines: list[str] = []
    lines.append(f"KEYWORD: {req.keyword!r}")
    lines.append(f"UPDATE: {update}")
    lines.append(f"WINDOW: before {report.before_date or '?'} -> after {report.after_date or '?'}")
    lines.append(f"CHURN: {report.churn_pct:.0f}% of the top-N changed")

    if report.n1_flip:
        lines.append(f"#1 FLIP: {report.n1_before or '?'} -> {report.n1_after or '?'}")
    else:
        lines.append(f"#1 HELD: {report.n1_after or report.n1_before or '?'}")

    if report.winners:
        lines.append("")
        lines.append("WINNERS:")
        for m in report.winners[:8]:
            lines.append("- " + _winrow(m))
    if report.losers:
        lines.append("")
        lines.append("LOSERS:")
        for m in report.losers[:8]:
            lines.append("- " + _winrow(m))

    if report.macro:
        lines.append("")
        lines.append("MACRO (mega-platform aggregate — brand/UGC giants, context only):")
        for k, v in report.macro.items():
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append(
        "WRITE THE ANALYSIS with these sections (use ## headings):\n"
        f"1. **What {update} Did to This SERP** — 2-3 sentences on the observed reshuffle.\n"
        "2. **Authority-Driven or Behaviour-Driven?** — judge whether the movement "
        "looks like a site-authority re-rating (incumbents/high-authority domains "
        "rewarded) or a per-query relevance/behaviour re-rating (better intent match "
        "winning regardless of authority). Cite the winners/losers to justify it.\n"
        "3. **What a Site Should Do** — concrete, honest next steps given the pattern.\n"
        "4. **Caveats** — third-party authority/traffic metrics are soft proxies, this "
        "is one keyword's national top-N, snapshots have cadence limits.\n"
        "Target ~500-700 words. Reason in the per-query-satisfaction frame. No hype."
    )
    return "\n".join(lines)


async def narrate_analyze(report: AnalyzeReport, settings=None) -> str:
    """Generate the markdown narrative for an analyze report. Never raises."""
    try:
        messages = [
            {"role": "system", "content": _ANALYZE_SYSTEM},
            {"role": "user", "content": _analyze_prompt(report)},
        ]
        text = await llm.chat(
            messages, model=None, max_tokens=2200,
            temperature=0.4, settings=settings,
        )
        if llm.llm_unavailable(text):
            return _FALLBACK
        return text.strip()
    except Exception:  # noqa: BLE001 — narrative is best-effort
        return _FALLBACK


async def narrate_compare(report: CompareReport, settings=None) -> str:
    """Generate the markdown narrative for a compare report. Never raises."""
    try:
        messages = [
            {"role": "system", "content": _COMPARE_SYSTEM},
            {"role": "user", "content": _compare_prompt(report)},
        ]
        text = await llm.chat(
            messages, model=None, max_tokens=2200,
            temperature=0.45, settings=settings,
        )
        if llm.llm_unavailable(text):
            return _FALLBACK
        return text.strip()
    except Exception:  # noqa: BLE001 — narrative is best-effort
        return _FALLBACK
