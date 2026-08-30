"""Orchestration — ties the clients, extraction, analysis, and report layers
into two end-to-end async flows: `run_analyze` and `run_compare`.

Pure async, no web framework. The CLI calls these via ``asyncio.run``; the API
calls them from a background task. Same code path, same `AnalyzeReport` /
`CompareReport` result.
"""
from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from ranklens.analyze.brand import enrich_brand
from ranklens.analyze.compare import build_compare
from ranklens.analyze.correlate import correlate, critical_value
from ranklens.analyze.engagement import analyze_engagement
from ranklens.analyze.entity_table import build_entity_table
from ranklens.analyze.funnel import build_competitor_cards, build_funnel
from ranklens.analyze.intent import analyze_intent
from ranklens.analyze.quality import analyze_quality
from ranklens.analyze.semantic import analyze_semantic
from ranklens.analyze.writing_brief import build_writing_brief
from ranklens.analyze.offpage import enrich_offpage
from ranklens.analyze.recommend import recommend
from ranklens.analyze.topical_authority import analyze_topical_authority
from ranklens.clients.authority import domain_authority
from ranklens.clients.crux import crux_metrics
from ranklens.extract.trust import trust_factors
from ranklens.clients.dataforseo import historical_serp, live_serp_advanced
from ranklens.clients.entities import canonicalize_attributes, extract_entities_llm
from ranklens.clients.fetch import fetch_pages
from ranklens.clients.serp import fetch_serp
from ranklens.config import get_settings, settings_for_analyze
from ranklens.extract.corpus import build_corpus, corpus_factors
from ranklens.extract.entities import apply_attribute_map, build_page_entities, entity_factors
from ranklens.extract.factors import extract_html_factors
from ranklens.extract.serp_factors import serp_factors
from ranklens.keywords import build_variation_set
from ranklens.models import (
    AnalyzeReport,
    AnalyzeRequest,
    CompareReport,
    CompareRequest,
    PageEntities,
    PageFactors,
    blank_request_byok,
)

# Rough per-call cost of the entity-extraction LLM pass (cheap router model).
_ENTITY_CALL_USD = 0.0012
from ranklens.report.ai import narrate_analyze, narrate_compare


def _domain(url: str) -> str:
    net = urlparse(url).netloc.lower()
    return net[4:] if net.startswith("www.") else net


async def _entity_pass(
    request: AnalyzeRequest,
    settings,
    page_factors: list[PageFactors],
    page_bodies: dict[int, str],
    fetched: dict,
    target_factors: dict | None,
    target_body: str | None,
):
    """Run the LLM entity/EAV layer and return the populated ``EntityTable``.

    For the top-N fetched ranking pages (and the tracked target), extract
    entities + attribute/value triples + edges with one concurrent LLM call
    each, merge the per-page entity factors into ``page_factors`` /
    ``target_factors`` in place, then build the topical EAV comparison table and
    inject the per-page ``EAV_COMPLETENESS`` factor for the correlation layer.
    Never raises — the caller wraps it, and every page degrades to empty.
    """
    top_n = max(1, settings.ranklens_entity_top_n)
    # The top-N fetched pages with body text, in SERP (rank) order.
    cand = [(idx, page_bodies[idx]) for idx in range(len(page_factors)) if idx in page_bodies][:top_n]

    sem = asyncio.Semaphore(max(1, settings.ranklens_entity_concurrency))

    async def _extract(idx: int, body: str):
        async with sem:
            raw = await extract_entities_llm(body, request.keyword, settings=settings)
        return idx, raw

    tasks = [_extract(idx, body) for idx, body in cand]
    if request.target_url and target_body:
        tasks.append(_extract(-1, target_body))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    n_calls = len(tasks)

    def _html_of(url: str) -> str:
        fp = fetched.get(url)
        return (fp.html if fp and getattr(fp, "html", None) else "") or ""

    page_entities: list[PageEntities] = []
    target_entities = None
    for res in results:
        if isinstance(res, Exception):
            continue
        idx, raw = res
        if idx == -1:
            html = _html_of(request.target_url or "")
            tpe = build_page_entities(
                raw, html, target_body or "", request.target_url or "", 0,
                _domain(request.target_url or ""),
            )
            target_entities = tpe
            if target_factors is not None:
                target_factors.update(entity_factors(tpe, html, target_body or ""))
        else:
            pf = page_factors[idx]
            html = _html_of(pf.url)
            pe = build_page_entities(raw, html, page_bodies[idx], pf.url, pf.rank, pf.domain)
            page_entities.append(pe)
            pf.factors.update(entity_factors(pe, html, page_bodies[idx]))

    # The target read is the one extraction the whole gap analysis hinges on —
    # a silent empty result grades the page against gaps it doesn't have
    # (observed live 2026-07-11, run ed8259a2cff3: page states phone+price,
    # extraction flapped to {}, report said "state your phone"). Retry it
    # sequentially before accepting an empty read.
    if (
        request.target_url
        and target_body
        and len(target_body) >= 300
        and (target_entities is None
             or (not target_entities.entities and not target_entities.triples))
    ):
        for _ in range(2):
            raw = await extract_entities_llm(target_body, request.keyword, settings=settings)
            n_calls += 1
            if raw and (raw.get("entities") or raw.get("triples")):
                html = _html_of(request.target_url)
                target_entities = build_page_entities(
                    raw, html, target_body, request.target_url, 0,
                    _domain(request.target_url),
                )
                if target_factors is not None:
                    target_factors.update(entity_factors(target_entities, html, target_body))
                break

    # Canonicalize attribute names across ALL pages (brands + target) into one
    # shared vocabulary BEFORE building the table, so near-synonyms
    # (cost_per_tooth / average_cost / dental_implant_price -> price;
    # miami_address / boca_raton_address -> address) collapse into a single
    # comparable row instead of fragmenting the attribute view. One extra LLM
    # call over the union of attributes; degrades to a no-op (identity) on
    # failure, so the table is unchanged when the canonicalizer is unavailable.
    canon_calls = 0
    try:
        attr_union: list[str] = []
        seen_attr: set[str] = set()
        for pe in page_entities + ([target_entities] if target_entities else []):
            for t in pe.triples or []:
                if getattr(t, "is_edge", False):
                    continue
                a = t.attribute
                if a and a not in seen_attr:
                    seen_attr.add(a)
                    attr_union.append(a)
        if len(attr_union) >= 2:
            attr_map = await canonicalize_attributes(
                attr_union, request.keyword, settings=settings
            )
            if attr_map:
                canon_calls = 1
                for pe in page_entities:
                    apply_attribute_map(pe, attr_map)
                if target_entities is not None:
                    apply_attribute_map(target_entities, attr_map)
    except Exception:
        canon_calls = 0

    table = build_entity_table(page_entities, target_entities, request.keyword, top_n=top_n)
    table.cost_usd = round((n_calls + canon_calls) * _ENTITY_CALL_USD, 4)

    # Per-page EAV completeness (covered pairs / union) -> a correlation factor.
    # Graded (2+ page consensus) rows only, so one page's idiosyncratic claims
    # can't drag every other page's completeness toward zero.
    rows = [r for r in (table.eav_rows or []) if getattr(r, "graded", False)]
    if rows:
        denom = float(len(rows))
        covered: dict[int, int] = {}
        for row in rows:
            for rank, cell in row.cells.items():
                if cell.present:
                    covered[rank] = covered.get(rank, 0) + 1
        for pf in page_factors:
            c = covered.get(pf.rank)
            if c is not None:
                pf.factors["EAV_COMPLETENESS"] = c / denom * 100.0
        if target_factors is not None and 0 in covered:
            target_factors["EAV_COMPLETENESS"] = covered[0] / denom * 100.0

    return table


# --------------------------------------------------------------------------- #
# ANALYZE — live Cora-style on-page correlation
# --------------------------------------------------------------------------- #
async def run_analyze(request: AnalyzeRequest, with_ai: bool = True) -> AnalyzeReport:
    settings = settings_for_analyze(request)
    # Secrets live on the per-run Settings copy only. Drop them from the request
    # so AnalyzeReport.request (and anything that dumps it) never carries them.
    blank_request_byok(request)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    max_pages = request.max_pages or settings.ranklens_max_pages
    vset = build_variation_set(request.keyword)

    # 1) SERP. The configured SERP provider only returns one page (~10 organic)
    #    — too few for a stable correlation. "auto" tries it, then falls back
    #    to DataForSEO live advanced (full depth) when that source is too shallow.
    ts = time.perf_counter()
    serp_cost = 0.0
    serp = None
    if request.serp_source in ("auto", "serpmaster"):
        try:
            serp = await fetch_serp(request.keyword, country=request.country, num=max_pages, settings=settings)
        except Exception:
            serp = None
    need_deep = request.serp_source == "dataforseo" or (
        request.serp_source == "auto" and (serp is None or len(serp.items) < 25)
    )
    if need_deep:
        try:
            deep, serp_cost = await live_serp_advanced(
                request.keyword, depth=max_pages, country=request.country,
                language=request.language, settings=settings,
            )
            if deep.items and (serp is None or len(deep.items) >= len(serp.items)):
                serp = deep
        except Exception:
            if serp is None:
                raise
    serp.language = request.language
    serp.items = serp.items[:max_pages]
    timings["serp_ms"] = (time.perf_counter() - ts) * 1000

    # 2) Fetch ranking pages (+ the target page if it isn't already ranked)
    serp_urls = [it.url for it in serp.items]
    fetch_list = list(serp_urls)
    target_in_serp_rank = None
    if request.target_url:
        tdom = _domain(request.target_url)
        for it in serp.items:
            if it.domain == tdom:
                target_in_serp_rank = it.rank
                break
        if request.target_url not in fetch_list:
            fetch_list.append(request.target_url)

    ts = time.perf_counter()
    fetched = await fetch_pages(fetch_list, concurrency=settings.ranklens_fetch_concurrency, settings=settings)
    timings["fetch_ms"] = (time.perf_counter() - ts) * 1000

    # 3) Per-page HTML factors (+ SERP factors), collecting body text for the corpus
    ts = time.perf_counter()
    page_factors: list[PageFactors] = []
    body_texts: list[str] = []
    page_bodies: dict[int, str] = {}  # index in page_factors -> body text
    for it in serp.items:
        fp = fetched.get(it.url)
        pf = PageFactors(url=it.url, rank=it.rank, domain=it.domain)
        # SERP-derived factors are always available
        try:
            pf.factors.update(serp_factors(it, vset))
        except Exception:
            pass
        if fp and fp.ok and fp.html:
            pf.fetched_ok = True
            pf.status_code = fp.status_code
            pf.load_ms = fp.load_ms
            try:
                html_factors, body = extract_html_factors(fp.html, it.url, it.domain, vset, fp.load_ms)
                pf.factors.update(html_factors)
                page_bodies[len(page_factors)] = body
                body_texts.append(body)
            except Exception as e:  # pragma: no cover - resilience
                pf.error = f"extract: {e}"
            try:
                pf.factors.update(trust_factors(fp.html, it.url))
            except Exception:
                pass
        else:
            pf.status_code = fp.status_code if fp else None
            pf.error = (fp.error if fp else "not fetched") or "fetch failed"
        page_factors.append(pf)

    # 4) Corpus pass (LSI / TF-IDF) over the fetched bodies
    corpus = build_corpus(body_texts, vset)
    for idx, body in page_bodies.items():
        try:
            page_factors[idx].factors.update(corpus_factors(body, vset, corpus))
        except Exception:
            pass
    timings["extract_ms"] = (time.perf_counter() - ts) * 1000

    # 5) Target factors (the tracked URL the user wants graded)
    target_factors = None
    target_body: str | None = None
    if request.target_url:
        tfp = fetched.get(request.target_url)
        if tfp and tfp.ok and tfp.html:
            try:
                tf, tbody = extract_html_factors(
                    tfp.html, request.target_url, _domain(request.target_url), vset, tfp.load_ms
                )
                target_body = tbody
                target_factors = dict(tf)
                target_factors.update(corpus_factors(tbody, vset, corpus))
                # SERP factors for the target if it ranks
                for it in serp.items:
                    if it.domain == _domain(request.target_url):
                        target_factors.update(serp_factors(it, vset))
                        break
                target_factors["__url__"] = request.target_url  # type: ignore[assignment]
                try:
                    target_factors.update(trust_factors(tfp.html, request.target_url))
                except Exception:
                    pass
            except Exception:
                target_factors = None

    # 5b) Off-page backlink enrichment (backlink provider) — page/domain/homepage authority
    #     become correlation factors; the target gets a backlink-quality panel.
    #     Opt-in via include_backlinks; degrades to a no-op without proxies.
    offpage_panel = None
    if request.include_backlinks:
        ts = time.perf_counter()
        try:
            per_url_bl, offpage_panel = await enrich_offpage(
                serp.items, request.target_url, request.country, settings=settings,
            )
        except Exception:
            per_url_bl, offpage_panel = {}, None
        for pf in page_factors:
            fx = per_url_bl.get(pf.url)
            if fx:
                pf.factors.update(fx)
        if target_factors is not None and request.target_url:
            tfx = per_url_bl.get(request.target_url)
            if tfx:
                target_factors.update(tfx)
        timings["backlinks_ms"] = (time.perf_counter() - ts) * 1000

    # 5c) Brand-demand enrichment (ranked-keywords) — branded search
    #     volume per ranking domain becomes a BRAND_VOLUME correlation factor;
    #     the target gets a brand-demand panel. Opt-in via include_brand;
    #     degrades to a no-op without proxies.
    brand_panel = None
    if request.include_brand:
        ts = time.perf_counter()
        try:
            per_url_brand, brand_panel = await enrich_brand(
                serp.items, request.target_url, request.country, settings=settings,
            )
        except Exception:
            per_url_brand, brand_panel = {}, None
        for pf in page_factors:
            fx = per_url_brand.get(pf.url)
            if fx:
                pf.factors.update(fx)
        if target_factors is not None and request.target_url:
            tfx = per_url_brand.get(request.target_url)
            if tfx:
                target_factors.update(tfx)
        timings["brand_ms"] = (time.perf_counter() - ts) * 1000

    # 5d) Entity / EAV enrichment (LLM) — discover topical entities + their
    #     attribute/value claims on the top-N ranking pages, emit per-page entity
    #     factors, and build the top-N topical EAV comparison table. Opt-in via
    #     include_entities; degrades to a no-op without an LLM key / on failure.
    entity_table = None
    if request.include_entities:
        ts = time.perf_counter()
        try:
            entity_table = await _entity_pass(
                request, settings, page_factors, page_bodies, fetched,
                target_factors, target_body,
            )
        except Exception:
            entity_table = None
        timings["entities_ms"] = (time.perf_counter() - ts) * 1000

    # 5e) Topical-authority enrichment — does the TARGET domain have a supporting
    #     content cluster for this topic, or is the page an island? Joins the
    #     site's own sitemap with a Google ``site:`` relevance query (both free,
    #     internal). Only runs with a target_url (it needs a domain); on by
    #     default via include_topical; degrades to None on any failure.
    topical_report = None
    if request.include_topical and request.target_url:
        ts = time.perf_counter()
        try:
            topical_report = await analyze_topical_authority(
                domain=request.target_url,
                topic=request.keyword,
                target_url=request.target_url,
                country=request.country,
                settings=settings,
                with_ai=with_ai,
            )
        except Exception:
            topical_report = None
        timings["topical_ms"] = (time.perf_counter() - ts) * 1000

    # 5f) Ranking-funnel panels — semantic passage coverage, intent/format fit,
    #     quality/effort rubric, simulated engagement, and CrUX field data. All
    #     independent, so they run concurrently; each degrades to None alone.
    #     Runs BEFORE correlation so the new per-page factors join the pool.
    #     Opt-out via include_funnel=False (--no-funnel): skip the whole block
    #     and leave funnel fields empty/None.
    semantic_report = intent_fit = quality_report = engagement_report = None
    if request.include_funnel:
        ts = time.perf_counter()
        try:
            bodies_by_rank: dict[int, str] = {
                page_factors[idx].rank: body for idx, body in page_bodies.items()
            }
            if target_body:
                bodies_by_rank[0] = target_body

            def _origin(url: str) -> str:
                p = urlparse(url)
                return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else ""

            origin_by_rank = {it.rank: _origin(it.url) for it in serp.items}
            if request.target_url:
                origin_by_rank[0] = _origin(request.target_url)
            origins = sorted({o for o in origin_by_rank.values() if o})

            # Two LLM analyzers at a time: the fixed-plan providers rate-limit hard
            # under 4-way concurrency (observed 429s in prod, 2026-07-11). CrUX is
            # not an LLM call and runs alongside everything.
            crux_task = asyncio.ensure_future(crux_metrics(origins, settings))
            sem_r, int_r = await asyncio.gather(
                analyze_semantic(request.keyword, serp.items, bodies_by_rank, settings),
                analyze_intent(request.keyword, serp.items, bodies_by_rank, request.target_url, settings),
                return_exceptions=True,
            )
            qual_r, eng_r = await asyncio.gather(
                analyze_quality(request.keyword, serp.items, bodies_by_rank, settings),
                analyze_engagement(request.keyword, serp.items, bodies_by_rank, request.target_url, settings),
                return_exceptions=True,
            )
            try:
                crux_r = await crux_task
            except Exception as e:  # noqa: BLE001 — CrUX is optional
                crux_r = e

            def _merge(factors_by_rank: dict[int, dict[str, float]]) -> None:
                for pf in page_factors:
                    fx = factors_by_rank.get(pf.rank)
                    if fx:
                        pf.factors.update(fx)
                if target_factors is not None and 0 in factors_by_rank:
                    target_factors.update(factors_by_rank[0])

            if not isinstance(sem_r, BaseException) and sem_r is not None:
                semantic_report, sem_fx = sem_r
                _merge(sem_fx or {})
            if not isinstance(int_r, BaseException):
                intent_fit = int_r
            if not isinstance(qual_r, BaseException) and qual_r is not None:
                quality_report, qual_fx = qual_r
                _merge(qual_fx or {})
            if not isinstance(eng_r, BaseException) and eng_r is not None:
                engagement_report, eng_fx = eng_r
                _merge(eng_fx or {})
            if not isinstance(crux_r, BaseException) and crux_r:
                _merge({rank: crux_r[origin] for rank, origin in origin_by_rank.items()
                        if origin in crux_r})
        except Exception:
            pass
        timings["funnel_panels_ms"] = (time.perf_counter() - ts) * 1000

    # 6) Correlate + recommend
    ts = time.perf_counter()
    fetched_ok_pages = [p for p in page_factors if p.fetched_ok]
    n = len(fetched_ok_pages) or len(page_factors)
    correlations = correlate(page_factors, target_factors)
    recommendations, target_summary = recommend(correlations, target_factors)
    if target_summary is not None:
        target_summary.url = request.target_url or target_summary.url
        target_summary.found_in_serp = target_in_serp_rank is not None
        target_summary.serp_rank = target_in_serp_rank
    timings["analyze_ms"] = (time.perf_counter() - ts) * 1000

    report = AnalyzeReport(
        request=request,
        serp=serp,
        page_factors=page_factors,
        correlations=correlations,
        recommendations=recommendations,
        target=target_summary,
        n_pages_analyzed=n,
        significance_threshold=critical_value(n),
        pages_fetched_ok=len(fetched_ok_pages),
        timings_ms=timings,
        cost_usd=round(serp_cost, 4),
    )
    report.offpage = offpage_panel
    if offpage_panel is not None:
        report.cost_usd = round(report.cost_usd + offpage_panel.cost_usd, 4)
    report.brand = brand_panel
    if brand_panel is not None:
        report.cost_usd = round(report.cost_usd + brand_panel.cost_usd, 4)
    report.entity_table = entity_table
    if entity_table is not None:
        report.cost_usd = round(report.cost_usd + entity_table.cost_usd, 4)
    report.topical = topical_report
    if topical_report is not None:
        report.cost_usd = round(report.cost_usd + topical_report.cost_usd, 4)
    report.semantic = semantic_report
    report.intent_fit = intent_fit
    report.quality = quality_report
    report.engagement = engagement_report
    for panel in (semantic_report, intent_fit, quality_report, engagement_report):
        if panel is not None:
            report.cost_usd = round(report.cost_usd + getattr(panel, "cost_usd", 0.0), 4)

    # 6a) The ranking funnel — the staged verdict over everything above, plus
    #     the per-competitor "why they outrank you" cards. Deterministic.
    #     Skipped entirely when include_funnel is False (fields stay None/[]).
    if request.include_funnel:
        report.funnel = build_funnel(report)
        report.competitor_cards = build_competitor_cards(report)
        # A 0 optimization score (no significant factor, or none met) renders as a
        # meaningless "Score 0" downstream — the funnel composite is the honest
        # replacement whenever the legacy score carries no signal.
        if (
            report.target is not None
            and report.funnel is not None
            and report.funnel.overall_score is not None
            and report.target.optimization_score <= 0
        ):
            report.target.optimization_score = report.funnel.overall_score

    # 6b) Writing brief — entity/EAV + roadmap gaps as concrete writing
    #     recommendations (deterministic, no LLM, free). None when there is
    #     nothing actionable.
    report.writing_brief = build_writing_brief(report)

    # 7) AI narrative
    if with_ai:
        ts = time.perf_counter()
        report.ai_narrative = await narrate_analyze(report, settings=settings)
        timings["ai_ms"] = (time.perf_counter() - ts) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return report


# --------------------------------------------------------------------------- #
# COMPARE — before/after an algorithm update
# --------------------------------------------------------------------------- #
async def run_compare(request: CompareRequest, with_ai: bool = True) -> CompareReport:
    settings = get_settings()
    cost = 0.0

    # BEFORE — historical dated SERP (DataForSEO is the only source of a past SERP)
    before, c1 = await historical_serp(
        request.keyword, request.update_date, country=request.country,
        language=request.language, num=request.depth, settings=settings,
    )
    cost += c1

    # AFTER — live SERP now (same source/geo for a clean diff)
    after, c2 = await live_serp_advanced(
        request.keyword, depth=request.depth, country=request.country,
        language=request.language, settings=settings,
    )
    cost += c2

    # Optional authority/traffic join
    authority = None
    if request.include_authority:
        domains = sorted({it.domain for it in before.items} | {it.domain for it in after.items})
        authority = await domain_authority(domains, settings=settings)

    report = build_compare(before, after, request, authority)
    report.cost_usd = round(cost, 4)

    if with_ai:
        report.ai_narrative = await narrate_compare(report, settings=settings)

    return report
