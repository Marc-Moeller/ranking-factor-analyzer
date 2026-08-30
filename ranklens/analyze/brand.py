"""Brand-demand enrichment — turns ranked-keyword data into a factor + panel.

Two jobs in one async pass, mirroring :mod:`ranklens.analyze.offpage`:

1. **A per-URL correlation factor.** For each unique domain in the SERP (+ the
   target's domain) we pull its organic ranked keywords and sum the monthly
   search volume of the BRANDED ones — the phrases that contain the domain's
   brand-name tokens. That branded-volume number is projected onto the
   ``BRAND_VOLUME`` factor id for every SERP URL on that domain, so the
   correlation layer can rank brand demand against position alongside the
   on-page factors.

2. **A :class:`BrandPanel` for the report.** The target's brand term + branded
   volume, the Page-1 average branded volume (top-N domains), and a one-line
   summary.

Data-source decision: ``/v1/keyword-research`` returns ``total_available`` but
an empty ``keywords[]``, so the brand signal is the summed search volume of
the domain's BRANDED ranked keywords from ``/v1/ranked-keywords`` instead.

Everything degrades gracefully: no proxies (``ranked_keywords`` returns ``[]``)
-> ``({}, None)``. The function never raises — risky parts are wrapped and the
report renders without the brand layer rather than failing the whole run.
"""
from __future__ import annotations

import re
from statistics import mean
from urllib.parse import urlsplit

from ranklens.clients.backlinks import ranked_keywords
from ranklens.config import Settings, get_settings
from ranklens.models import BrandCompetitor, BrandKeyword, BrandPanel

# Brand factor id (must match ranklens.factors_registry).
BRAND_VOLUME = "BRAND_VOLUME"

# Caps to keep the brand pass cheap: one failover call per domain, bounded set.
MAX_BRAND_DOMAINS = 20
# Pull deep: a domain that ranks poorly (e.g. #11) for its own high-volume brand
# terms has low traffic on them, so they sit far down a traffic-sorted list. 100
# rows missed "nuvia dental implant center" (18.1k) entirely — 1000 surfaces it.
RANKED_KW_LIMIT = 1000
TOP_N_FOR_AVERAGES = 10

# Display caps for the brand comparison panel.
MAX_TARGET_VARIATIONS = 25      # rows in the "your brand searches" table
MAX_COMPETITORS = 5             # rival domains in the leaderboard (top SERP-ranked)
MAX_COMPETITOR_VARIATIONS = 4   # variations shown per competitor row

# Social networks, directories and aggregators are NOT brand competitors — their
# branded volume (facebook = 100M+) is colossal and irrelevant to a local/niche
# comparison, so they're dropped from the leaderboard and the page-1 average.
NON_COMPETITOR_SLDS = frozenset({
    "facebook", "instagram", "youtube", "twitter", "x", "linkedin", "tiktok",
    "pinterest", "reddit", "yelp", "tripadvisor", "wikipedia", "google", "bing",
    "yahoo", "amazon", "ebay", "quora", "medium", "threads", "snapchat",
    "whatsapp", "glassdoor", "indeed", "nextdoor", "angi", "thumbtack",
    "groupon", "foursquare", "mapquest", "bbb", "healthgrades", "zocdoc",
    "webmd", "vitals", "ratemds", "wellness", "yellowpages",
})


def _is_competitor(domain: str) -> bool:
    """True unless ``domain`` is a social/directory/aggregator (not a real rival)."""
    return _sld_label(domain).lower() not in NON_COMPETITOR_SLDS


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


def _domain_of(url: str) -> str:
    """www-stripped host of ``url`` (best-effort, never raises)."""
    try:
        host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _sld_label(domain: str) -> str:
    """The registrable SLD label of a domain.

    Strips the public-suffix tail so ``exampledentalclinic.com.au`` ->
    ``exampledentalclinic`` and ``examplestore.com`` -> ``examplestore``. Best-effort: handles the
    common ``*.co.uk`` / ``*.com.au`` two-part suffixes without a PSL dependency.
    """
    parts = [p for p in (domain or "").lower().split(".") if p]
    if not parts:
        return ""
    # Two-part public suffixes we care about (au/uk/etc).
    second_level = {"com", "co", "net", "org", "gov", "edu", "ac"}
    if len(parts) >= 3 and parts[-2] in second_level and len(parts[-1]) == 2:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _brand_term(domain: str) -> str:
    """Humanized brand term for display: split the SLD into words.

    Splits on hyphens/underscores and on lowercase->uppercase boundaries, then
    lowercases and collapses whitespace. A glued lowercase SLD like
    ``exampledentalclinic`` has no boundaries to split on, so it stays as one token
    for *display* — matching against ranked keywords uses the collapsed-alnum
    form below, which still recognizes the spaced "city dental rooms" phrases.
    """
    sld = _sld_label(domain)
    if not sld:
        return ""
    # camelCase boundary -> space.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", sld)
    s = re.sub(r"[-_]+", " ", s)
    return " ".join(s.lower().split())


def _collapse_alnum(text: str) -> str:
    """Lowercase ``text`` and strip everything that isn't a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _words(text: str) -> list[str]:
    """Whitespace tokens of ``text``, each collapsed to lowercase alnum (drops empties)."""
    return [w for w in (_collapse_alnum(t) for t in (text or "").lower().split()) if w]


def _brand_core(rows: list[dict], sld: str) -> str:
    """Derive the short brand root when the domain glues it to a descriptor.

    Many brands register a domain that appends a descriptor to the real brand
    name — ``brightsmiles.com`` for the brand "Nuvia", ``smiledirectclub.com`` for
    "SmileDirectClub". The collapsed SLD (``brightsmiles``) then fails to match the
    brand's actual searches ("nuvia dental implant center", "nuvia", ...).

    We recover the root from the keyword corpus itself: the brand core is the
    SHORTEST **single-word** keyword that (a) is a prefix of the collapsed SLD,
    (b) is ≥5 chars, and (c) people search as a STANDALONE term (the bare brand
    name). Shortest wins because the root must be general enough to also lead the
    longer brand phrases ("nuvia" leads "nuvia dental implant center"); the
    bare-search + ≥5-char prefix requirement keeps short generic fragments out.
    Returns ``""`` when no such root exists (the SLD already is the brand).

    Known limitation: a descriptive-geo SLD whose leading word is itself a
    searched term (``city`` in ``exampledentalclinic``) could be recovered as a
    "brand", inflating its branded volume with generic "city ..." terms. The
    panel is opt-in context and degrades gracefully, so this is accepted.
    """
    if not sld:
        return ""
    best = ""
    for row in rows:
        raw = str(row.get("phrase") or "").strip()
        if len(raw.split()) != 1:      # bare brand name = a single word
            continue
        ph = _collapse_alnum(raw)
        if len(ph) >= 5 and ph != sld and sld.startswith(ph):
            if best == "" or len(ph) < len(best):
                best = ph
    return best


def _branded_rows(rows: list[dict], domain: str) -> tuple[list[dict], float, str] | None:
    """Pull a domain's BRANDED ranked keywords (the brand-search variations).

    A keyword is "branded" when its collapsed-alnum phrase CONTAINS the domain's
    collapsed-alnum SLD label (catches glued "exampledentalclinic", hyphenated, and
    spaced "city dental rooms ... reviews" variants) OR one of its words equals
    the derived brand core (catches "nuvia dental implant center", "nuvia", ...
    when the domain is ``brightsmiles``). See :func:`_brand_core`.

    Returns ``(variations, total, brand_core)`` where ``variations`` is the
    deduped list of ``{"phrase","volume"}`` branded rows sorted by descending
    volume, ``total`` is their summed monthly search volume, and ``brand_core``
    is the recovered brand root (or ``""``). Returns ``None`` when no branded
    keyword is found (so a domain with zero brand demand reads as "no data"
    rather than a hard zero that would skew the correlation).
    """
    sld = _collapse_alnum(_sld_label(domain))
    if not sld or not rows:
        return None
    core = _brand_core(rows, sld)
    by_phrase: dict[str, float] = {}
    for row in rows:
        raw = str(row.get("phrase") or "").strip()
        if not raw:
            continue
        collapsed = _collapse_alnum(raw)
        words = _words(raw)
        # Branded = the phrase carries the glued SLD, or it LEADS with the brand
        # root ("nuvia ...", "nuvia"). Leading-word (not any-word) keeps generic
        # phrases that merely happen to contain the root out.
        branded = sld in collapsed or (core != "" and words[:1] == [core])
        if not branded:
            continue
        vol = _num(row.get("volume"))
        if vol is None:
            continue
        # Keep the largest volume seen for a given phrase (dedupe variants).
        key = raw.lower()
        if vol >= by_phrase.get(key, -1.0):
            by_phrase[key] = vol
    if not by_phrase:
        return None
    variations = sorted(
        ({"phrase": p, "volume": v} for p, v in by_phrase.items()),
        key=lambda r: r["volume"],
        reverse=True,
    )
    total = float(sum(r["volume"] for r in variations))
    return variations, total, core


def _branded_volume(rows: list[dict], domain: str) -> float | None:
    """Summed branded search volume for ``domain`` (thin wrapper over rows)."""
    res = _branded_rows(rows, domain)
    return res[1] if res is not None else None


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
async def enrich_brand(
    serp_items: list,
    target_url: str | None,
    country: str,
    settings: Settings | None = None,
) -> tuple[dict[str, dict], "BrandPanel | None"]:
    """Enrich a SERP with branded search volume: a factor + a panel.

    Args:
        serp_items: the ranking set (``list[SerpItem]``).
        target_url: optional tracked URL — its domain's brand term and branded
            volume go in the panel.
        country: Google ``gl`` code -> provider database code (e.g. ``au``).
        settings: optional pre-loaded settings.

    Returns:
        ``(per_url_factors, panel)`` where ``per_url_factors`` maps each SERP URL
        (and ``target_url`` when supplied) to ``{BRAND_VOLUME: float}``, and
        ``panel`` is a :class:`BrandPanel` (or ``None`` when no branded-volume
        data was available — e.g. no proxies configured).
    """
    settings = settings or get_settings()

    # 1) Unique domains to query (de-duped, capped). The target's domain is
    #    included even if it isn't in the SERP.
    domains: list[str] = []
    seen: set[str] = set()

    def _add(dom: str) -> None:
        if dom and dom not in seen:
            seen.add(dom)
            domains.append(dom)

    for it in serp_items:
        _add(getattr(it, "domain", "") or _domain_of(getattr(it, "url", "")))

    target_domain = ""
    if target_url:
        target_domain = _domain_of(target_url)
        _add(target_domain)

    domains = domains[:MAX_BRAND_DOMAINS]
    if not domains:
        return {}, None

    # Best (lowest) SERP position seen per domain — used to pick "top 5
    # competitors" and to label each leaderboard row.
    domain_rank: dict[str, int] = {}
    for it in serp_items:
        dom = getattr(it, "domain", "") or _domain_of(getattr(it, "url", ""))
        r = getattr(it, "rank", None)
        if dom and isinstance(r, int):
            if dom not in domain_rank or r < domain_rank[dom]:
                domain_rank[dom] = r

    # 2) One ranked-keywords call per domain. Degrade gracefully on any failure.
    domain_volume: dict[str, float] = {}
    domain_variations: dict[str, list[dict]] = {}
    domain_core: dict[str, str] = {}
    for dom in domains:
        try:
            rows = await ranked_keywords(
                dom, country, limit=RANKED_KW_LIMIT, settings=settings
            )
        except Exception:  # noqa: BLE001 — brand is opt-in context, never fatal
            continue
        res = _branded_rows(rows, dom)
        if res is not None:
            variations, total, core = res
            domain_volume[dom] = total
            domain_variations[dom] = variations
            if core:
                domain_core[dom] = core

    if not domain_volume:
        return {}, None

    # 3) Project each domain's brand volume onto every SERP URL on that domain.
    per_url_factors: dict[str, dict] = {}
    for it in serp_items:
        dom = getattr(it, "domain", "") or _domain_of(getattr(it, "url", ""))
        vol = domain_volume.get(dom)
        if vol is not None:
            per_url_factors[getattr(it, "url", "")] = {BRAND_VOLUME: vol}

    # Inject the target's factor too (even if it isn't in the SERP).
    if target_url and target_domain in domain_volume:
        per_url_factors[target_url] = {BRAND_VOLUME: domain_volume[target_domain]}

    # 4) Build the report panel. ------------------------------------------- #
    panel = BrandPanel(cost_usd=0.0, domain_brand_volume=dict(domain_volume))

    def _display_brand(dom: str) -> str:
        """Prefer the recovered brand root (``nuvia``) over the glued SLD."""
        return domain_core.get(dom) or _brand_term(dom)

    if target_url and target_domain:
        panel.brand_term = _display_brand(target_domain)
        panel.target_brand_volume = domain_volume.get(target_domain)

    # Page-1 average branded volume over the top-N ranked domains with data.
    try:
        ranked = sorted(serp_items, key=lambda x: x.rank)[:TOP_N_FOR_AVERAGES]
        vals: list[float] = []
        seen_dom: set[str] = set()
        for it in ranked:
            dom = getattr(it, "domain", "") or _domain_of(getattr(it, "url", ""))
            if dom in seen_dom or not _is_competitor(dom):
                continue
            seen_dom.add(dom)
            v = domain_volume.get(dom)
            if v is not None:
                vals.append(v)
        if vals:
            panel.page1_avg_brand_volume = round(mean(vals), 1)
    except Exception:  # noqa: BLE001 — averages are best-effort
        pass

    # 4b) Brand-variation breakdown + competitor leaderboard. -------------- #
    def _competitor(dom: str) -> BrandCompetitor:
        variations = domain_variations.get(dom, [])
        return BrandCompetitor(
            domain=dom,
            brand_term=_display_brand(dom),
            total_volume=domain_volume.get(dom, 0.0),
            keyword_count=len(variations),
            rank=domain_rank.get(dom),
            top_keywords=[
                BrandKeyword(phrase=r["phrase"], volume=r["volume"])
                for r in variations[:MAX_COMPETITOR_VARIATIONS]
            ],
            is_target=bool(target_domain) and dom == target_domain,
        )

    # The target's own brand variations (full list, capped for display).
    if target_domain and target_domain in domain_variations:
        panel.target_brand_keywords = [
            BrandKeyword(phrase=r["phrase"], volume=r["volume"])
            for r in domain_variations[target_domain][:MAX_TARGET_VARIATIONS]
        ]

    # Pick the top-N competitors by SERP position (best-ranked rivals first),
    # excluding the target's own domain and any social/directory aggregators.
    # Domains absent from the SERP sort last.
    rival_domains = [
        d for d in domain_volume if d != target_domain and _is_competitor(d)
    ]
    rival_domains.sort(key=lambda d: (domain_rank.get(d, 10_000), -domain_volume[d]))
    chosen = rival_domains[:MAX_COMPETITORS]
    if target_domain and target_domain in domain_volume:
        chosen.append(target_domain)

    # Leaderboard ordered by branded search volume (strongest brand on top).
    leaderboard = sorted(
        (_competitor(d) for d in chosen),
        key=lambda c: c.total_volume,
        reverse=True,
    )
    panel.competitors = leaderboard
    for i, c in enumerate(leaderboard, start=1):
        if c.is_target:
            panel.brand_rank = i
            break

    # 5) One-line summary. -------------------------------------------------- #
    tv = panel.target_brand_volume
    avg = panel.page1_avg_brand_volume
    if panel.brand_term and tv is not None:
        bits = [f'brand "{panel.brand_term}" — {int(round(tv)):,} branded searches/mo']
        if panel.brand_rank is not None and len(leaderboard) > 1:
            bits.append(f"#{panel.brand_rank} of {len(leaderboard)} brands")
        if avg is not None:
            bits.append(f"page-1 avg {int(round(avg)):,}")
        panel.summary = " · ".join(bits)
    elif domain_volume:
        panel.summary = f"branded search volume read for {len(domain_volume)} ranking domains"

    return per_url_factors, panel
