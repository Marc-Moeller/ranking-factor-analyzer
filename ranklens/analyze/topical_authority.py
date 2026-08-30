"""Topical-authority analyzer — does a DOMAIN have a supporting content cluster
for a page's topic?

The question this answers: when ``example.com/dental-implants/`` ranks (or wants
to), is the rest of the site *about* dental implants too — a real topical cluster —
or is that page an island? Cora can't see this; it's pure on-page. This joins two
independent evidence sources:

1. **Sitemap inventory** (the site's own claim). Discover + expand the sitemap
   (``ranklens.clients.sitemap``), then match each URL's slug against an
   LLM-expanded variation set. High precision, complete, free of Google's caps.

2. **Google ``site:`` relevance** (Google's association). One
   ``site:domain (v1 | v2 | ...)`` query through the configured SERP provider. Google caps ``site:``
   at ~10 results regardless of ``num`` (measured), so we don't trust the *count* —
   we use the *returned pages*, re-filtered against the variation set by
   title+slug to drop noise (homepage / blog index leak in otherwise).

The cluster is the **union**. The score blends breadth (how many on-topic pages),
focus (what share of the site is this topic), supporting-subtopic coverage, and
indexation. An LLM writes the plain-English verdict. Everything degrades: no
sitemap -> SERP-only; LLM hiccup -> a deterministic fallback variation set.
"""
from __future__ import annotations

import json
import re
import time
from urllib.parse import urlsplit

from ranklens.clients import llm
from ranklens.clients.serp import fetch_serp
from ranklens.clients.sitemap import fetch_inventory
from ranklens.config import Settings, get_settings
from ranklens.models import TopicalAuthorityReport, TopicalPage

# Tuning knobs (transparent, not magic — see _score).
TARGET_CLUSTER = 8          # this many on-topic pages = full breadth credit
FOCUS_FULL = 0.15           # ≥15% of the site on-topic = full focus credit
INDEX_FULL = 3              # ≥3 genuine site: hits = full indexation credit
MAX_SITE_QUERY_TERMS = 12   # cap OR-piped terms so the site: query stays sane
SUPPORT_MIN_CLUSTER = 3     # cluster this deep is enough to "support" a target page

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "at",
    "by", "from", "is", "it", "as", "your", "you", "best", "top", "near", "me",
}
_WORD_RE = re.compile(r"[a-z0-9]+")
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# Term matching — slug/title text vs the variation set, stem-aware
# --------------------------------------------------------------------------- #
def _content_tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


def _term_pattern(term: str) -> re.Pattern | None:
    """A stem-aware regex for one variation.

    Multi-word terms become an ordered sequence with flexible separators
    (``dental implant`` matches ``dental-implants`` in a slug). Each content token
    is matched as a stem prefix (``implant`` matches ``implants``, ``implantology``)
    so we don't miss morphological variants. Returns ``None`` for an all-stopword
    term.
    """
    toks = _content_tokens(term)
    if not toks:
        return None
    # \bimplant\w*  (stem prefix), joined by 1-3 non-word chars / short glue words.
    parts = [re.escape(t) + r"\w*" for t in toks]
    pattern = r"\b" + r"(?:\W+\w*?){0,2}?".join(parts) if len(parts) > 1 else r"\b" + parts[0]
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _slug_text(url: str) -> str:
    """Humanized path text for slug matching (``/dental-implants/sydney/`` ->
    ``dental implants sydney``)."""
    try:
        path = urlsplit(url).path
    except ValueError:
        path = url
    path = re.sub(r"\.(html?|php|aspx?)$", "", path, flags=re.IGNORECASE)
    return re.sub(r"[-_/]+", " ", path).strip().lower()


def _domain_of(url: str) -> str:
    try:
        host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _norm_url(url: str) -> str:
    """Loose canonical form for union/dedup across sitemap + SERP (scheme/www/
    trailing-slash insensitive)."""
    try:
        s = urlsplit(url if "://" in url else f"https://{url}")
    except ValueError:
        return url.rstrip("/").lower()
    host = (s.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = s.path.rstrip("/") or "/"
    return f"{host}{path}".lower()


class _Matcher:
    """Compiled term set: which core/variation/adjacent terms a text hits."""

    def __init__(self, core_terms: list[str], adjacent_terms: list[str]) -> None:
        self.core = [(t, p) for t in core_terms if (p := _term_pattern(t))]
        self.adjacent = [(t, p) for t in adjacent_terms if (p := _term_pattern(t))]

    def match_core(self, text: str) -> list[str]:
        low = text.lower()
        return [t for t, p in self.core if p.search(low)]

    def match_adjacent(self, text: str) -> list[str]:
        low = text.lower()
        return [t for t, p in self.adjacent if p.search(low)]


# --------------------------------------------------------------------------- #
# LLM topic expansion (with a deterministic fallback)
# --------------------------------------------------------------------------- #
def _fallback_expand(topic: str) -> dict:
    """Cheap deterministic variation set if the LLM is unavailable.

    Deliberately conservative: a multi-word topic keeps the full phrase plus only
    its most-specific (last) content token — never the generic head modifier on
    its own ("dental implants" -> phrase + "implants", NOT bare "dental", which
    would over-match every dental page). Single-word topics keep the token.
    """
    toks = _content_tokens(topic)
    core = " ".join(toks) or topic.strip().lower()
    variations = [core]
    if len(toks) >= 2:
        variations.append(toks[-1])      # most specific noun only
    elif toks:
        variations.append(toks[0])
    return {"core": core, "variations": list(dict.fromkeys(variations)), "adjacent": []}


# How many times to retry the router when it 503/overloads (it intermittently does).
_LLM_RETRIES = 3


async def _chat_json(messages: list[dict], settings: Settings, max_tokens: int) -> dict | None:
    """Run a chat completion expected to return a JSON object, with retries.

    The free LLM router intermittently returns 503 (overloaded) — ``llm.chat``
    surfaces that as an ``"[AI report unavailable...]"`` string. We retry a few
    times, then parse the first JSON object out of the reply. Returns ``None`` if
    every attempt failed or no JSON could be parsed.
    """
    for attempt in range(_LLM_RETRIES):
        reply = await llm.chat(messages, max_tokens=max_tokens, temperature=0.3, settings=settings)
        if llm.llm_unavailable(reply):
            # Missing-key sentinel will never recover; skip retries.
            if reply and reply.startswith("[AI unavailable:"):
                return None
            continue  # transient router error — retry
        match = _JSON_OBJ_RE.search(reply)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            continue  # truncated/garbled JSON — retry for a clean one
    return None


async def expand_topic(topic: str, settings: Settings) -> dict:
    """LLM-expand ``topic`` into ``{core, variations[], adjacent[]}``.

    ``variations`` are same-topic synonyms / word-stem forms (used for matching +
    the site: query); ``adjacent`` are supporting subtopics that build authority
    but are NOT the same topic. Falls back to :func:`_fallback_expand` on any
    failure so the analyzer always has a term set.
    """
    system = (
        "You are an SEO topical-authority analyst. Given a page's core topic, you "
        "produce the cluster of search-term variations, synonyms and word-stem "
        "forms a search engine treats as the SAME topic, plus closely-related "
        "supporting subtopics."
    )
    user = (
        f'Core topic: "{topic}"\n\n'
        "Return ONLY a JSON object with:\n"
        '- "core": the canonical 1-3 word topic phrase\n'
        '- "variations": 8-15 search-term variations / synonyms / close word-stem '
        "forms meaning the SAME topic (no location modifiers, no brand names)\n"
        '- "adjacent": 3-6 closely related supporting subtopics that build topical '
        "authority but are NOT the same topic\n\n"
        'Example for "dental implants": '
        '{"core":"dental implants","variations":["tooth implant","teeth implants",'
        '"implant dentistry","tooth replacement"],"adjacent":["bone graft",'
        '"all on 4","implant cost"]}'
    )
    try:
        obj = await _chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            settings=settings,
            max_tokens=1000,
        )
        if not obj:
            return _fallback_expand(topic)
        core = (obj.get("core") or topic).strip()
        variations = [str(v).strip() for v in (obj.get("variations") or []) if str(v).strip()]
        adjacent = [str(v).strip() for v in (obj.get("adjacent") or []) if str(v).strip()]
        # Always include the core phrase in the variation set.
        if core and core.lower() not in [v.lower() for v in variations]:
            variations.insert(0, core)
        if not variations:
            return _fallback_expand(topic)
        return {"core": core, "variations": variations, "adjacent": adjacent}
    except Exception:  # noqa: BLE001 — expansion is best-effort
        return _fallback_expand(topic)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score(cluster_size: int, sitemap_total: int, adjacent_covered: int,
           n_adjacent_terms: int, serp_hits: int) -> tuple[float, float]:
    """Return ``(score_0_100, focus_ratio)`` from the cluster signals.

    Transparent weighted blend with diminishing returns on each component:
      breadth 45% · focus 20% · adjacent coverage 20% · indexation 15%.
    """
    breadth = min(1.0, cluster_size / TARGET_CLUSTER)
    focus_ratio = (cluster_size / sitemap_total) if sitemap_total else 0.0
    focus = min(1.0, focus_ratio / FOCUS_FULL) if sitemap_total else 0.0
    adjacent = (adjacent_covered / n_adjacent_terms) if n_adjacent_terms else 0.0
    indexation = min(1.0, serp_hits / INDEX_FULL)
    score = 100.0 * (0.45 * breadth + 0.20 * focus + 0.20 * adjacent + 0.15 * indexation)
    return round(score, 1), round(focus_ratio, 4)


def _band(score: float) -> str:
    if score >= 70:
        return "strong"
    if score >= 40:
        return "moderate"
    return "thin"


# --------------------------------------------------------------------------- #
# AI verdict (optional, best-effort)
# --------------------------------------------------------------------------- #
async def _ai_verdict(report: TopicalAuthorityReport, settings: Settings) -> str | None:
    sample = [p.url for p in report.cluster[:12]]
    listing = "\n".join(f"- {u}" for u in sample) or "(none found)"
    system = (
        "You are an SEO topical-authority analyst. You judge whether a domain has a "
        "genuine supporting content cluster for a topic. Be concise and grounded in "
        "the numbers given — no fluff, no invented facts."
    )
    user = (
        f'Domain: {report.domain}\nTopic: "{report.topic}"\n'
        f"Target page: {report.target_url or '(none)'}\n\n"
        f"Sitemap pages total: {report.sitemap_total}\n"
        f"On-topic cluster pages: {report.cluster_size} "
        f"({report.focus_ratio*100:.1f}% of the site)\n"
        f"Supporting subtopics covered: {report.adjacent_covered}/{len(report.adjacent)}\n"
        f"Google site: relevant hits: {report.serp_indexed_hits}\n"
        f"Topical-authority score: {report.score}/100 ({report.band})\n"
        f"Target page in cluster: {report.target_in_cluster}\n\n"
        f"On-topic pages found:\n{listing}\n\n"
        "In 2-4 sentences: does this domain have the topical authority to support "
        "rankings for this topic/page, or is the page an island? If thin, name the "
        "single most useful gap to fill."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(_LLM_RETRIES):
        try:
            reply = await llm.chat(msgs, max_tokens=400, temperature=0.4, settings=settings)
        except Exception:  # noqa: BLE001
            continue
        if not llm.llm_unavailable(reply):
            return reply
    return None


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
async def analyze_topical_authority(
    domain: str,
    topic: str | None = None,
    target_url: str | None = None,
    country: str = "us",
    settings: Settings | None = None,
    with_ai: bool = True,
) -> TopicalAuthorityReport:
    """Measure a domain's topical authority for a topic.

    Args:
        domain: the domain to analyze (bare host or URL).
        topic: the topic to score. If ``None``, derived from ``target_url``'s slug.
        target_url: optional specific page we're checking the cluster supports.
        country: Google ``gl`` code for the site: query.
        settings: optional pre-loaded settings.
        with_ai: run the LLM verdict narrative (the variation expansion always
            tries the LLM regardless, with a deterministic fallback).

    Returns:
        A :class:`TopicalAuthorityReport`. Never raises — missing sitemap or SERP
        signals degrade the score rather than failing the run.
    """
    settings = settings or get_settings()
    t0 = time.perf_counter()

    dom = _domain_of(domain) or domain.strip().lower()
    if not topic:
        topic = _topic_from_url(target_url) if target_url else dom.split(".")[0]

    # 1) LLM term expansion. ------------------------------------------------- #
    terms = await expand_topic(topic, settings)
    core, variations, adjacent = terms["core"], terms["variations"], terms["adjacent"]
    matcher = _Matcher(variations, adjacent)

    # 2) Sitemap inventory. -------------------------------------------------- #
    try:
        _sitemaps, pages = await fetch_inventory(dom)
    except Exception:  # noqa: BLE001 — sitemap is one signal, not the run
        pages = []
    sitemap_found = bool(pages)

    # Accumulate cluster pages keyed by normalized URL (union across sources).
    cluster: dict[str, TopicalPage] = {}

    def _touch(url: str) -> TopicalPage:
        key = _norm_url(url)
        tp = cluster.get(key)
        if tp is None:
            tp = TopicalPage(url=url)
            cluster[key] = tp
        return tp

    for url in pages:
        slug = _slug_text(url)
        core_hits = matcher.match_core(slug)
        adj_hits = matcher.match_adjacent(slug)
        if core_hits or adj_hits:
            tp = _touch(url)
            tp.in_sitemap = True
            tp.slug_match = bool(core_hits)
            if core_hits:
                tp.matched_terms = sorted(set(tp.matched_terms) | set(core_hits))
            else:
                tp.is_adjacent = True
                tp.matched_terms = sorted(set(tp.matched_terms) | set(adj_hits))

    # 3) Google site: relevance query (variations OR-piped). ----------------- #
    site_terms = variations[:MAX_SITE_QUERY_TERMS]
    or_block = " | ".join(f'"{t}"' for t in site_terms)
    serp_query = f"site:{dom} {or_block}".strip()
    serp_hits = 0
    try:
        serp = await fetch_serp(serp_query, country=country, num=20, settings=settings)
    except Exception:  # noqa: BLE001 — SERP is one signal
        serp = None
    canonical_norm: str | None = None
    if serp and serp.items:
        for i, it in enumerate(serp.items):
            # Re-filter: only count results that genuinely match a core variation
            # in title or slug (drops homepage / blog-index noise).
            text = f"{it.title} {_slug_text(it.url)}"
            core_hits = matcher.match_core(text)
            if not core_hits:
                continue
            serp_hits += 1
            tp = _touch(it.url)
            tp.serp_hit = True
            tp.title = it.title
            tp.matched_terms = sorted(set(tp.matched_terms) | set(core_hits))
            if canonical_norm is None:
                canonical_norm = _norm_url(it.url)

    # 4) Assemble cluster + counts. ----------------------------------------- #
    cluster_pages = list(cluster.values())
    core_pages = [p for p in cluster_pages if not p.is_adjacent or p.slug_match or p.serp_hit]
    # A page is "core cluster" if it matched a core variation anywhere.
    core_cluster = [p for p in cluster_pages if p.slug_match or p.serp_hit]
    adjacent_pages = [p for p in cluster_pages if p.is_adjacent and not (p.slug_match or p.serp_hit)]

    cluster_size = len(core_cluster)
    adjacent_covered = len({t for p in adjacent_pages for t in p.matched_terms})

    # 5) Target verdict. ----------------------------------------------------- #
    target_in_cluster = False
    target_is_canonical = False
    if target_url:
        tnorm = _norm_url(target_url)
        target_in_cluster = tnorm in cluster
        target_is_canonical = canonical_norm is not None and tnorm == canonical_norm

    # 6) Score + bands. ------------------------------------------------------ #
    score, focus_ratio = _score(
        cluster_size, len(pages), adjacent_covered, len(adjacent), serp_hits
    )
    band = _band(score) if (sitemap_found or serp_hits) else "unknown"
    supports_target = cluster_size >= SUPPORT_MIN_CLUSTER and (
        target_in_cluster if target_url else True
    )

    # Order cluster: core slug+serp first, then serp, then slug, then adjacent.
    def _rank(p: TopicalPage) -> tuple:
        return (
            0 if (p.slug_match and p.serp_hit) else 1 if p.serp_hit else 2 if p.slug_match else 3,
            p.url,
        )

    cluster_sorted = sorted(cluster_pages, key=_rank)

    summary = (
        f"{cluster_size} on-topic page(s) of {len(pages)} "
        f"({focus_ratio*100:.0f}% focus) · "
        f"{adjacent_covered}/{len(adjacent)} supporting subtopics · "
        f"{serp_hits} Google-relevant · score {score}/100 ({band})"
    )

    report = TopicalAuthorityReport(
        domain=dom,
        topic=topic,
        country=country,
        target_url=target_url,
        core=core,
        variations=variations,
        adjacent=adjacent,
        sitemap_total=len(pages),
        sitemap_found=sitemap_found,
        cluster=cluster_sorted,
        cluster_size=cluster_size,
        adjacent_pages=len(adjacent_pages),
        adjacent_covered=adjacent_covered,
        serp_indexed_hits=serp_hits,
        target_in_cluster=target_in_cluster,
        target_is_canonical=target_is_canonical,
        focus_ratio=focus_ratio,
        score=score,
        band=band,
        supports_target=supports_target,
        summary=summary,
    )

    if with_ai:
        report.ai_narrative = await _ai_verdict(report, settings)

    report.cost_usd = 0.0  # internal SERP + router both free-tier
    _ = time.perf_counter() - t0
    return report


def _topic_from_url(target_url: str) -> str:
    """Humanized last meaningful slug of a URL as a topic hint."""
    try:
        parts = [p for p in urlsplit(target_url).path.split("/") if p]
    except ValueError:
        parts = []
    for seg in reversed(parts):
        cleaned = re.sub(r"\.(html?|php|aspx?)$", "", seg, flags=re.IGNORECASE)
        cleaned = re.sub(r"[-_]+", " ", cleaned).strip()
        if cleaned and not cleaned.isdigit():
            return cleaned
    dom = _domain_of(target_url)
    return re.sub(r"[-_]+", " ", dom.split(".")[0]).strip() if dom else "the page topic"
