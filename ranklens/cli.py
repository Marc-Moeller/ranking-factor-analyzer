"""RankLens CLI.

    python -m ranklens.cli analyze "cordless impact driver" --url https://example.com/drivers --pages 20 --open
    python -m ranklens.cli compare "cordless impact driver" --date 2026-05-21 --name "May 2026 Core Update" --open

Runs the pipeline, writes a self-contained HTML report under data/reports/, and
persists the run to the SQLite store.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import uuid
import webbrowser
from datetime import datetime, timezone

# Windows consoles default to cp1252 and choke on arrows/bullets — force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ranklens.config import get_settings
from ranklens.models import (
    AnalyzeRequest,
    CompareRequest,
    Run,
    RunKind,
    RunStatus,
)
from ranklens.pipeline import run_analyze, run_compare
from ranklens.report.html import render_analyze, render_compare, save_report
from ranklens.store import save_run


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:50] or "report"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _persist(kind: RunKind, label: str, request: dict, result_obj, status: RunStatus, error: str | None) -> Run:
    run = Run(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        status=status,
        label=label,
        request=request,
        result=result_obj.model_dump(mode="json") if result_obj is not None else None,
        error=error,
        finished_at=datetime.now(timezone.utc),
    )
    save_run(run)
    return run


def cmd_analyze(args) -> None:
    settings = get_settings()
    req = AnalyzeRequest(
        keyword=args.keyword,
        target_url=args.url,
        country=args.country or settings.ranklens_default_country,
        language=args.language,
        max_pages=args.pages,
        include_authority=args.authority,
        include_backlinks=args.backlinks,
        include_brand=args.brand,
        include_entities=args.entities,
        include_funnel=not args.no_funnel,
    )
    print(f"→ Analyzing SERP for: {req.keyword!r}  (country={req.country}, pages={req.max_pages or settings.ranklens_max_pages})")
    report = asyncio.run(run_analyze(req, with_ai=not args.no_ai))

    html = render_analyze(report)
    out = settings.data_dir / "reports" / f"analyze_{_slug(req.keyword)}_{_stamp()}.html"
    save_report(html, out)
    run = _persist(RunKind.analyze, f'analyze: "{req.keyword}"', req.model_dump(mode="json"), report, RunStatus.done, None)

    print(f"\n  Pages fetched OK : {report.pages_fetched_ok}/{report.n_pages_analyzed}")
    print(f"  Significance |r| > {report.significance_threshold:.3f}")
    if report.target:
        print(f"  Optimization score: {report.target.optimization_score:.0f}/100  "
              f"({report.target.factors_met}/{report.target.factors_total} factors met, {report.target.quick_wins} quick wins)")
    if report.offpage:
        op = report.offpage
        ps = op.target_page_stats
        if ps and ps.authority_score is not None:
            print(f"  Off-page: page authority {ps.authority_score:.0f}"
                  f"{f' (page-1 avg {op.page1_avg_page_authority:.0f})' if op.page1_avg_page_authority is not None else ''}, "
                  f"{ps.referring_domains or 0} ref domains, {ps.total_backlinks or 0} backlinks")
        if op.link_quality and op.link_quality.score is not None:
            lq = op.link_quality
            print(f"  Link quality: {lq.score:.0f}/100  ({lq.sample_size} links scored)")
    if report.brand and report.brand.target_brand_volume is not None:
        br = report.brand
        avg = f" (page-1 avg {br.page1_avg_brand_volume:,.0f})" if br.page1_avg_brand_volume is not None else ""
        print(f"  Brand demand: \"{br.brand_term}\" {br.target_brand_volume:,.0f} branded searches/mo{avg}")
    if report.entity_table and report.entity_table.n_entities:
        et = report.entity_table
        comp = f", you cover {et.target_completeness*100:.0f}% of claims" if et.target_completeness is not None else ""
        miss = f", {len(et.must_add_entities)} entities missing" if et.must_add_entities else ""
        print(f"  Entities: {et.n_entities} topical / {et.n_pairs} attribute claims across top {len(et.ranks)}{comp}{miss}")
    print(f"\n  Top actions:")
    for r in report.recommendations[:6]:
        print(f"   • [{r.phase}] {r.name}: {r.action_text}")
    print(f"\n  Run id: {run.id}")
    print(f"  Report: {out}")
    if args.open:
        webbrowser.open(out.as_uri())


def cmd_compare(args) -> None:
    settings = get_settings()
    req = CompareRequest(
        keyword=args.keyword,
        update_date=args.date,
        update_name=args.name,
        country=args.country or settings.ranklens_default_country,
        language=args.language,
        depth=args.depth,
        include_authority=not args.no_authority,
    )
    print(f"→ Comparing SERP before/after {req.update_name or req.update_date} for: {req.keyword!r}")
    try:
        report = asyncio.run(run_compare(req, with_ai=not args.no_ai))
    except RuntimeError as e:
        print(f"\n  ✗ {e}")
        print("  Tip: historical SERP data is sparse for long-tail terms — try a broader head keyword.")
        _persist(RunKind.compare, f'compare: "{req.keyword}"', req.model_dump(mode="json"), None, RunStatus.error, str(e))
        return

    html = render_compare(report)
    out = settings.data_dir / "reports" / f"compare_{_slug(req.keyword)}_{_stamp()}.html"
    save_report(html, out)
    run = _persist(RunKind.compare, f'compare: "{req.keyword}"', req.model_dump(mode="json"), report, RunStatus.done, None)

    print(f"\n  Before {report.before_date} → After {report.after_date}")
    if report.n1_flip:
        print(f"  #1 FLIP: {report.n1_before} → {report.n1_after}")
    else:
        print(f"  #1 held: {report.n1_after}")
    print(f"  Winners: {', '.join(m.domain for m in report.winners[:6]) or '—'}")
    print(f"  Losers : {', '.join(m.domain for m in report.losers[:6]) or '—'}")
    print(f"  Churn  : {report.churn_pct:.0f}%   Cost: ${report.cost_usd:.3f}")
    print(f"\n  Run id: {run.id}")
    print(f"  Report: {out}")
    if args.open:
        webbrowser.open(out.as_uri())


def cmd_topical(args) -> None:
    from ranklens.analyze.topical_authority import analyze_topical_authority

    settings = get_settings()
    country = args.country or settings.ranklens_default_country
    print(f"→ Topical authority for {args.domain!r}"
          f"{f' · topic {args.topic!r}' if args.topic else ''}"
          f"{f' · target {args.url}' if args.url else ''}  (country={country})")

    report = asyncio.run(analyze_topical_authority(
        domain=args.domain,
        topic=args.topic,
        target_url=args.url,
        country=country,
        with_ai=not args.no_ai,
    ))

    print(f"\n  Topic        : {report.topic}  (core: {report.core})")
    print(f"  Variations   : {', '.join(report.variations[:10])}"
          f"{' …' if len(report.variations) > 10 else ''}")
    if report.adjacent:
        print(f"  Subtopics    : {', '.join(report.adjacent)}")
    sm = f"{report.sitemap_total} pages" if report.sitemap_found else "not found"
    print(f"  Sitemap      : {sm}")
    print(f"  Cluster      : {report.cluster_size} on-topic page(s) "
          f"({report.focus_ratio*100:.0f}% of site), "
          f"{report.adjacent_covered}/{len(report.adjacent)} supporting subtopics")
    print(f"  Google site: : {report.serp_indexed_hits} relevant hit(s)")
    if report.target_url:
        flag = "✓ in cluster" if report.target_in_cluster else "✗ NOT in cluster (island)"
        canon = " · canonical for topic" if report.target_is_canonical else ""
        print(f"  Target page  : {flag}{canon}")
    print(f"\n  SCORE        : {report.score}/100  [{report.band.upper()}]"
          f"  →  {'supports the target' if report.supports_target else 'does NOT support the target'}")
    print(f"\n  On-topic pages:")
    for p in report.cluster[:15]:
        tags = []
        if p.serp_hit:
            tags.append("google")
        if p.slug_match:
            tags.append("slug")
        if p.is_adjacent and not p.slug_match and not p.serp_hit:
            tags.append("subtopic")
        print(f"   • {p.url}  [{','.join(tags)}]")
    if report.ai_narrative:
        print(f"\n  AI verdict:\n  {report.ai_narrative}")

    if args.json:
        out = settings.data_dir / "reports" / f"topical_{_slug(report.domain)}_{_stamp()}.json"
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n  JSON: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ranklens", description="RankLens — Cora replacement: SERP factor analysis + algo-update comparison.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Live on-page correlation analysis for a keyword.")
    a.add_argument("keyword")
    a.add_argument("--url", help="Your target URL to grade against the SERP.")
    a.add_argument("--country", default=None, help="Google country code (us, au, gb, ...).")
    a.add_argument("--language", default="en")
    a.add_argument("--pages", type=int, default=None, help="How many ranking pages to analyze.")
    a.add_argument("--authority", action="store_true", help="Pull backlink/authority factors (needs an authority API key).")
    a.add_argument("--backlinks", action="store_true", help="Pull page/domain backlink power + a target link-quality panel.")
    a.add_argument("--brand", action="store_true", help="Pull branded search volume as a brand-demand factor + panel.")
    a.add_argument("--entities", action=argparse.BooleanOptionalAction, default=True,
                   help="LLM entity/EAV discovery + the top-N topical entity comparison table (on by default; --no-entities to skip).")
    a.add_argument("--no-funnel", action="store_true",
                   help="Skip ranking-funnel panels (semantic/intent/quality/engagement/CrUX).")
    a.add_argument("--no-ai", action="store_true", help="Skip the AI narrative.")
    a.add_argument("--open", action="store_true", help="Open the HTML report in a browser.")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("compare", help="Before/after an algorithm update for a keyword.")
    c.add_argument("keyword")
    c.add_argument("--date", required=True, help="Rollout start date (YYYY-MM-DD) = the 'before' cutoff.")
    c.add_argument("--name", default=None, help="Update name, e.g. 'May 2026 Core Update'.")
    c.add_argument("--country", default=None)
    c.add_argument("--language", default="en")
    c.add_argument("--depth", type=int, default=20)
    c.add_argument("--no-authority", action="store_true", help="Skip authority/traffic join.")
    c.add_argument("--no-ai", action="store_true")
    c.add_argument("--open", action="store_true")
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("topical", help="Does a domain have topical authority (a supporting content cluster) for a topic/page?")
    t.add_argument("domain", help="Domain to analyze (bare host or URL).")
    t.add_argument("--topic", default=None, help="Topic to score. Defaults to the --url slug, else the domain name.")
    t.add_argument("--url", default=None, help="A specific page to check the cluster supports.")
    t.add_argument("--country", default=None, help="Google country code for the site: query (us, au, gb, ...).")
    t.add_argument("--no-ai", action="store_true", help="Skip the AI verdict narrative.")
    t.add_argument("--json", action="store_true", help="Also write the full report JSON to data/reports/.")
    t.set_defaults(func=cmd_topical)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
