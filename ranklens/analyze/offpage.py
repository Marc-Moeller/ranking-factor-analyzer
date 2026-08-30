"""Off-page enrichment — turns backlink data into factors + a report panel.

Two jobs in one async pass:

1. **Per-URL correlation factors.** From ONE bulk call we get an authority
   score / referring-domain / backlink read on each ranking URL, its registrable
   domain, and the site homepage. We project those onto the off-page factor ids
   (``PAGE_AUTHORITY``, ``REF_DOMAINS``, ``HOMEPAGE_AUTHORITY``, …) so the
   correlation layer can rank them against position alongside the on-page factors.

2. **An :class:`OffPagePanel` for the report.** The tracked target's page /
   homepage / domain backlink stats, the Page-1 authority averages, and — when a
   target URL is supplied — its top inbound links graded for *topical relevance*
   by a single LLM pass (each source scored 0-1 against the page's topic, derived
   from the target URL slug + anchor texts).

Everything degrades gracefully: no proxies (``bulk_backlink_stats`` returns ``{}``)
-> ``({}, None)``; an LLM hiccup -> ``topical_relevance`` left ``None``. The
function never raises — risky parts are wrapped and the report renders without the
off-page layer rather than failing the whole run.
"""
from __future__ import annotations

import asyncio
import json
import re
from statistics import mean
from urllib.parse import urlsplit

from ranklens.clients.backlinks import bulk_backlink_stats, page_backlinks
from ranklens.clients import llm
from ranklens.config import Settings, get_settings
from ranklens.models import Backlink, BacklinkStats, LinkQuality, OffPagePanel

# Off-page factor ids (must match ranklens.factors_registry).
PAGE_AUTHORITY = "PAGE_AUTHORITY"
PAGE_REF_DOMAINS = "PAGE_REF_DOMAINS"
PAGE_BACKLINKS = "PAGE_BACKLINKS"
PAGE_FOLLOW_RATIO = "PAGE_FOLLOW_RATIO"
AUTHORITY_SCORE = "AUTHORITY_SCORE"
REF_DOMAINS = "REF_DOMAINS"
BACKLINKS = "BACKLINKS"
HOMEPAGE_AUTHORITY = "HOMEPAGE_AUTHORITY"

# Caps to keep one bulk call cheap and one LLM call within a sane token budget.
MAX_BULK_TARGETS = 200
MAX_RELEVANCE_LINKS = 25
PAGE_BACKLINK_LIMIT = 50          # target page: deeper sample (drives link quality)
COMPETITOR_BACKLINK_LIMIT = 25    # each other ranking site: lighter sample
BACKLINK_SITES_TOP_N = 10         # how many ranking sites to pull link lists for
TOP_N_FOR_AVERAGES = 10

# Extracts the first JSON array (e.g. ``[0.1, 0.8, ...]``) from an LLM reply.
_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


# --------------------------------------------------------------------------- #
# Small numeric / string helpers
# --------------------------------------------------------------------------- #
def _num(value) -> float | None:
    """Coerce ``value`` to ``float``; return ``None`` on missing/garbage input."""
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _homepage_for(domain: str) -> str:
    """The canonical homepage URL for a registrable domain."""
    return f"https://{domain.strip().strip('/')}/"


def _domain_of(url: str) -> str:
    """www-stripped host of ``url`` (best-effort, never raises)."""
    try:
        host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _dofollow_ratio(follow: float | None, nofollow: float | None) -> float | None:
    """``follow / (follow + nofollow)`` as a 0-1 ratio; ``None`` if undefined."""
    f = follow or 0.0
    nf = nofollow or 0.0
    denom = f + nf
    if denom <= 0:
        return None
    return f / denom


def _stats_from_row(row: dict | None, scope: str, target: str) -> BacklinkStats | None:
    """Build a :class:`BacklinkStats` from one bulk row."""
    if not row:
        return None
    follow = _num(row.get("follow"))
    nofollow = _num(row.get("nofollow"))
    ref = _num(row.get("referring_domains"))
    total = _num(row.get("total_backlinks"))
    return BacklinkStats(
        scope=scope,
        target=target,
        authority_score=_num(row.get("authority_score")),
        referring_domains=int(ref) if ref is not None else None,
        total_backlinks=int(total) if total is not None else None,
        follow=int(follow) if follow is not None else None,
        nofollow=int(nofollow) if nofollow is not None else None,
        dofollow_ratio=_dofollow_ratio(follow, nofollow),
    )


def _topic_from_url(target_url: str, backlinks: list[Backlink]) -> str:
    """Derive a short topic hint for the relevance prompt.

    Primary signal is the target URL's last meaningful path slug, humanized
    (``.../dental-implants-perth/`` -> ``dental implants perth``). Falls back to
    the registrable domain label when the path is empty (a homepage target).
    """
    try:
        parts = [p for p in urlsplit(target_url).path.split("/") if p]
    except ValueError:
        parts = []
    slug = ""
    for seg in reversed(parts):
        cleaned = re.sub(r"\.(html?|php|aspx?)$", "", seg, flags=re.IGNORECASE)
        cleaned = re.sub(r"[-_]+", " ", cleaned).strip()
        # Skip pure numbers / ids; we want a wordy slug.
        if cleaned and not cleaned.isdigit():
            slug = cleaned
            break
    if not slug:
        domain = _domain_of(target_url)
        label = domain.split(".")[0] if domain else ""
        slug = re.sub(r"[-_]+", " ", label).strip()
    return slug or "the target page's main topic"


# --------------------------------------------------------------------------- #
# LLM topical-relevance pass
# --------------------------------------------------------------------------- #
def _parse_relevance(text: str, n: int) -> list[float | None]:
    """Defensively parse a JSON array of 0-1 floats from the LLM reply.

    Returns a length-``n`` list (truncated/padded with ``None``). Any failure
    yields all-``None`` so callers degrade rather than crash.
    """
    out: list[float | None] = [None] * n
    if llm.llm_unavailable(text):
        return out
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return out
    try:
        arr = json.loads(match.group(0))
    except (ValueError, TypeError):
        return out
    if not isinstance(arr, list):
        return out
    for i in range(min(n, len(arr))):
        v = _num(arr[i])
        if v is None:
            continue
        out[i] = max(0.0, min(1.0, v))  # clamp to [0, 1]
    return out


async def _score_topical_relevance(
    backlinks: list[Backlink],
    topic: str,
    settings: Settings,
) -> None:
    """Mutate ``backlinks`` in place, setting ``topical_relevance`` via one LLM call."""
    sample = backlinks[:MAX_RELEVANCE_LINKS]
    if not sample:
        return

    lines = []
    for i, bl in enumerate(sample):
        anchor = (bl.anchor or "").strip()[:120]
        lines.append(
            f"{i}. source_domain={bl.source_domain or _domain_of(bl.source_url)} | "
            f"anchor={anchor!r} | url={bl.source_url}"
        )
    listing = "\n".join(lines)

    system = (
        "You are an SEO link-quality analyst. You judge how topically relevant "
        "a linking source is to a target page's topic. Relevance is about subject "
        "matter, not link power."
    )
    user = (
        f"Our target page's topic is: \"{topic}\".\n\n"
        f"Below are {len(sample)} sources linking to our page. For EACH one, rate "
        f"how topically relevant the linking source is to our topic on a 0-1 scale "
        f"(0 = unrelated, 0.5 = loosely related, 1 = squarely on-topic).\n\n"
        f"{listing}\n\n"
        f"Respond with ONLY a JSON array of {len(sample)} floats in the same order, "
        f"nothing else. Example: [0.9, 0.2, 0.7]"
    )

    try:
        reply = await llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=900,
            temperature=0.2,
            settings=settings,
        )
    except Exception:  # noqa: BLE001 — relevance is best-effort
        return

    scores = _parse_relevance(reply, len(sample))
    for bl, score in zip(sample, scores):
        bl.topical_relevance = score


# --------------------------------------------------------------------------- #
# LinkQuality composite
# --------------------------------------------------------------------------- #
def _link_quality(backlinks: list[Backlink]) -> LinkQuality | None:
    """Composite 0-100 quality read over a backlink sample.

    Blends three weighted components — source authority, topical relevance, and
    dofollow share — renormalizing the weights over whichever components are
    actually present so a missing signal doesn't drag the score to zero.
    """
    if not backlinks:
        return None

    n = len(backlinks)
    authorities = [bl.source_authority for bl in backlinks if bl.source_authority is not None]
    relevances = [bl.topical_relevance for bl in backlinks if bl.topical_relevance is not None]
    dofollow_count = sum(1 for bl in backlinks if bl.dofollow)

    avg_authority = mean(authorities) if authorities else None
    mean_relevance = mean(relevances) if relevances else None
    dofollow_ratio = dofollow_count / n if n else None

    # Weighted blend with graceful renormalization over available components.
    components: list[tuple[float, float]] = []  # (weight, normalized 0-1 value)
    if avg_authority is not None:
        components.append((0.4, max(0.0, min(1.0, avg_authority / 100.0))))
    if mean_relevance is not None:
        components.append((0.3, max(0.0, min(1.0, mean_relevance))))
    if dofollow_ratio is not None:
        components.append((0.3, max(0.0, min(1.0, dofollow_ratio))))

    score: float | None = None
    if components:
        total_w = sum(w for w, _ in components)
        if total_w > 0:
            score = 100.0 * sum(w * v for w, v in components) / total_w

    summary = (
        f"{n} sampled links · "
        f"avg source authority {round(avg_authority, 1) if avg_authority is not None else 'n/a'} · "
        f"mean relevance {round(mean_relevance, 2) if mean_relevance is not None else 'n/a'} · "
        f"dofollow {round(100 * dofollow_ratio) if dofollow_ratio is not None else 'n/a'}% · "
        f"composite {round(score) if score is not None else 'n/a'}/100"
    )

    return LinkQuality(
        score=round(score, 1) if score is not None else None,
        sample_size=n,
        avg_source_authority=round(avg_authority, 1) if avg_authority is not None else None,
        mean_topical_relevance=round(mean_relevance, 3) if mean_relevance is not None else None,
        dofollow_ratio=round(dofollow_ratio, 3) if dofollow_ratio is not None else None,
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Factor projection
# --------------------------------------------------------------------------- #
def _factors_for(
    url: str,
    domain: str,
    stats: dict[str, dict],
) -> dict[str, float]:
    """Project the url / domain / homepage rows onto off-page factor ids.

    Only keys with a present (non-``None``) value are included. All values float.
    """
    out: dict[str, float] = {}
    url_row = stats.get(url) or {}
    dom_row = stats.get(domain) or {}
    home_row = stats.get(_homepage_for(domain)) or {}

    # Page-level (the exact ranking URL).
    pa = _num(url_row.get("authority_score"))
    if pa is not None:
        out[PAGE_AUTHORITY] = pa
    prd = _num(url_row.get("referring_domains"))
    if prd is not None:
        out[PAGE_REF_DOMAINS] = prd
    pbl = _num(url_row.get("total_backlinks"))
    if pbl is not None:
        out[PAGE_BACKLINKS] = pbl

    follow_ratio = _dofollow_ratio(_num(url_row.get("follow")), _num(url_row.get("nofollow")))
    if follow_ratio is not None:
        out[PAGE_FOLLOW_RATIO] = follow_ratio * 100.0  # 0-100 percent

    # Domain-level (registrable root domain).
    da = _num(dom_row.get("authority_score"))
    if da is not None:
        out[AUTHORITY_SCORE] = da
    drd = _num(dom_row.get("referring_domains"))
    if drd is not None:
        out[REF_DOMAINS] = drd
    dbl = _num(dom_row.get("total_backlinks"))
    if dbl is not None:
        out[BACKLINKS] = dbl

    # Homepage authority.
    ha = _num(home_row.get("authority_score"))
    if ha is not None:
        out[HOMEPAGE_AUTHORITY] = ha

    return out


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
async def enrich_offpage(
    serp_items: list,
    target_url: str | None,
    country: str,
    settings: Settings | None = None,
    with_llm: bool = True,
) -> tuple[dict[str, dict], "OffPagePanel | None"]:
    """Enrich a SERP with off-page data: factors + a report panel.

    Args:
        serp_items: the ranking set (``list[SerpItem]``).
        target_url: optional tracked URL to grade (its page/homepage/domain
            stats and its inbound-link quality go in the panel).
        country: Google ``gl`` code (passed through for signature symmetry;
            backlink data is not geo-scoped).
        settings: optional pre-loaded settings.
        with_llm: when ``True`` (and a ``target_url`` is given), score the
            target's inbound links for topical relevance via one LLM call.

    Returns:
        ``(per_url_factors, panel)`` where ``per_url_factors`` maps each SERP URL
        (and ``target_url`` when supplied) to ``{factor_id: float}``, and
        ``panel`` is an :class:`OffPagePanel` (or ``None`` when no backlink data
        was available — e.g. no proxies configured).
    """
    settings = settings or get_settings()

    # 1) Build the bulk target list (de-duped, capped). -------------------- #
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(t: str, ttype: str) -> None:
        if not t:
            return
        key = (t, ttype)
        if key in seen:
            return
        seen.add(key)
        targets.append(key)

    for it in serp_items:
        _add(it.url, "url")
        _add(it.domain, "root_domain")
        _add(_homepage_for(it.domain), "url")

    target_domain = ""
    if target_url:
        target_domain = _domain_of(target_url)
        _add(target_url, "url")
        if target_domain:
            _add(target_domain, "root_domain")
            _add(_homepage_for(target_domain), "url")

    targets = targets[:MAX_BULK_TARGETS]
    if not targets:
        return {}, None

    # 2) ONE bulk call. Degrade gracefully on any failure / no proxies.
    try:
        stats = await bulk_backlink_stats(targets, settings=settings)
    except Exception:  # noqa: BLE001 — off-page is opt-in context, never fatal
        return {}, None
    if not stats:
        return {}, None

    # 3) Per-URL factor projection. ---------------------------------------- #
    per_url_factors: dict[str, dict] = {}
    for it in serp_items:
        factors = _factors_for(it.url, it.domain, stats)
        if factors:
            per_url_factors[it.url] = factors

    # Inject the target's factors too (even if it isn't in the SERP).
    if target_url:
        tfactors = _factors_for(target_url, target_domain, stats)
        if tfactors:
            per_url_factors[target_url] = tfactors

    # 4) Build the report panel. ------------------------------------------- #
    panel = OffPagePanel(cost_usd=0.0)

    # Target page / homepage / domain stats.
    if target_url:
        panel.target_page_stats = _stats_from_row(stats.get(target_url), "url", target_url)
        if target_domain:
            home = _homepage_for(target_domain)
            panel.target_homepage_stats = _stats_from_row(stats.get(home), "homepage", home)
            panel.target_domain_stats = _stats_from_row(
                stats.get(target_domain), "domain", target_domain
            )

    # Page-1 authority averages over the top-N ranked items that have data.
    try:
        ranked = sorted(serp_items, key=lambda x: x.rank)[:TOP_N_FOR_AVERAGES]
        pa_vals = [
            per_url_factors[it.url][PAGE_AUTHORITY]
            for it in ranked
            if it.url in per_url_factors and PAGE_AUTHORITY in per_url_factors[it.url]
        ]
        rd_vals = [
            per_url_factors[it.url][PAGE_REF_DOMAINS]
            for it in ranked
            if it.url in per_url_factors and PAGE_REF_DOMAINS in per_url_factors[it.url]
        ]
        if pa_vals:
            panel.page1_avg_page_authority = round(mean(pa_vals), 2)
        if rd_vals:
            panel.page1_avg_ref_domains = round(mean(rd_vals), 1)
    except Exception:  # noqa: BLE001 — averages are best-effort
        pass

    # 5) Backlink lists for the target + each top-N ranking site, tagged by
    #    destination so the report can filter "links to which site". The target
    #    page is sampled deeper (drives link quality + relevance); the other sites
    #    are sampled lighter. All fetched concurrently. -------------------- #
    ranked = sorted(serp_items, key=lambda x: x.rank)[:BACKLINK_SITES_TOP_N]

    # Fetch plan: (url, limit, to_domain, to_rank), de-duped by url so the target
    # (which may itself rank) is fetched once — at the deeper target limit.
    plan: list[tuple[str, int, str, int | None]] = []
    planned_urls: set[str] = set()
    if target_url:
        target_rank = next((it.rank for it in serp_items if it.url == target_url), None)
        plan.append((target_url, PAGE_BACKLINK_LIMIT,
                     target_domain or _domain_of(target_url), target_rank))
        planned_urls.add(target_url)
    for it in ranked:
        if it.url in planned_urls:
            continue
        planned_urls.add(it.url)
        plan.append((it.url, COMPETITOR_BACKLINK_LIMIT, it.domain, it.rank))

    async def _fetch(url: str, limit: int) -> dict:
        try:
            return await page_backlinks(url, target_type="url", limit=limit, settings=settings)
        except Exception:  # noqa: BLE001 — one site failing must not sink the rest
            return {}

    raw_results = (
        await asyncio.gather(*[_fetch(u, lim) for (u, lim, _, _) in plan]) if plan else []
    )

    competitor_backlinks: list[Backlink] = []
    target_backlinks: list[Backlink] = []
    for (url, _lim, to_dom, to_rank), raw in zip(plan, raw_results):
        site_links: list[Backlink] = []
        for b in (raw.get("backlinks") or []):
            source_url = b.get("source_url") or ""
            if not source_url:
                continue
            try:
                site_links.append(
                    Backlink(
                        source_url=source_url,
                        source_domain=b.get("source_domain") or _domain_of(source_url),
                        anchor=b.get("anchor") or "",
                        dofollow=bool(b.get("dofollow", True)),
                        source_authority=_num(b.get("source_authority")),
                        domain_authority=_num(b.get("domain_authority")),
                        first_seen=b.get("first_seen"),
                        to_domain=to_dom,
                        to_rank=to_rank,
                    )
                )
            except Exception:  # noqa: BLE001 — skip a malformed row, keep the rest
                continue
        if target_url and url == target_url:
            target_backlinks = site_links  # same objects also flow into the table
        competitor_backlinks.extend(site_links)

    # LLM topical relevance on the TARGET's links only (cost control). Because the
    # target rows are the same objects living in competitor_backlinks, the scores
    # show up there too without a second pass.
    if target_backlinks and with_llm:
        topic = _topic_from_url(target_url, target_backlinks)
        try:
            await _score_topical_relevance(target_backlinks, topic, settings)
        except Exception:  # noqa: BLE001 — relevance is best-effort
            pass

    # Table order: by SERP rank, then strongest source authority first.
    competitor_backlinks.sort(
        key=lambda bl: (bl.to_rank if bl.to_rank is not None else 999,
                        -(bl.source_authority or 0.0))
    )

    panel.target_backlinks = target_backlinks
    panel.competitor_backlinks = competitor_backlinks
    panel.link_quality = _link_quality(target_backlinks)

    return per_url_factors, panel
