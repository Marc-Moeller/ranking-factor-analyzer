"""Topical EAV comparison — turns per-page entity reads into the top-N gap table.

This is the analysis behind Cora's "topical EAV comparison". It takes the entity
/ EAV reads for the top ranking pages (the "brands") plus the tracked target
page and projects them onto a shared, consensus-derived grid so three kinds of
gap become explicit for the target:

1. **Missing entities** — an entity the brands talk about that the target never
   mentions at all.
2. **Missing attributes** — the target covers the entity but is silent on an
   attribute the brands assert.
3. **Missing / off-consensus values** — the target asserts the attribute, but
   with no value, or with a value that disagrees with the brand consensus.

It produces three matrices on one :class:`EntityTable`:

* **View A — ``coverage_rows``**: one row per distinct topical entity, with a
  present/absent + salience cell per page column.
* **View B — ``eav_rows``**: one row per ``(entity, attribute)`` attribute→value
  claim, with the asserted value per page column, a modal consensus value, and
  the per-row gap classification above.
* **View C — ``edge_rows``**: one row per ``(entity, relation)`` entity→entity
  connection (``is_edge=True``), the per-page cell holding the linked target
  entity, with the same consensus + gap treatment.

Consensus is built over the BRAND pages only (the target is graded *against* it,
never *part* of it). Entities are aligned by ``canonical_id`` (falling back to a
local normalization of the surface name); EAV rows by
``(canonical_entity, normalized_attribute)``; edges by ``(canonical_entity,
attribute)`` with ``is_edge=True``.

Three alignment layers make cross-page consensus real on live SERPs (live-run
audit 2026-07-11: without them, 137/137 EAV rows had consensus=1 — every page
talks about *its own* business/products, so raw pairs never align):

* **Entity alias merge** (:func:`_build_alias_map`) — brand-prefix expansions
  and identical distinctive cores collapse onto one canon ("Aeron Chair" /
  "Herman Miller Aeron" / "Herman Miller Aeron Chair" become one row).
  Person/location entities never merge.
* **First-party subject role** — each page's OWN business (domain-brand match,
  or the entity carrying its contact facts) and the keyword-topic entity are
  remapped to one shared ``__subject__`` row for the attribute matrices, so
  "the page's own price / address / phone / experience" align across pages
  instead of fragmenting into per-business rows.
* **Attribute families** (:func:`_attr_family`) — a deterministic fallback that
  collapses obvious families (``member_11_30min_rate`` → ``price``) even when
  the LLM corpus canonicalizer flaps.

Gap grading is **consensus-gated**: a row must be asserted by at least
:data:`MIN_GAP_CONSENSUS` brand pages before the target is graded against it
(``EavRow.graded``). Single-page claims stay visible as market intel but are
never counted as "your gaps", and ``target_completeness`` is computed over
graded rows only.

Everything degrades gracefully: an empty/degenerate input (``pages=[]``) returns
a valid empty :class:`EntityTable`. The function never raises — risky parts are
wrapped and the report renders without the entity table rather than failing the
whole run. The LLM cost of *discovering* the entities is accounted upstream in
the pipeline, so ``cost_usd`` is always ``0.0`` here.
"""
from __future__ import annotations

import re
from collections import Counter

from ranklens.models import EavRow, EntityCell, EntityTable, PageEntities

# Row caps — keep pathological pages from blowing up the table. The HTML layer
# does its own display capping on top of these.
MAX_COVERAGE_ROWS = 60
MAX_EAV_ROWS = 400
MAX_MUST_ADD = 25

# A row must be asserted by at least this many brand pages before the target is
# graded against it. Below the gate the row is still shown (market intel) but
# carries no gap and does not count toward completeness — one competitor's
# private phone number is not "your gap".
MIN_GAP_CONSENSUS = 2

# Numeric value match tolerance (±15%).
VALUE_TOLERANCE = 0.15

# First number in a string (handles "$10", "10.5", "1,200", "10%").
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# The shared first-party row: each page's own business/service claims are
# remapped here so they align across pages (see module docstring).
SUBJECT_KEY = "__subject__"
SUBJECT_DISPLAY = "The ranking business"
SUBJECT_TYPE = "first party"


# --------------------------------------------------------------------------- #
# Small string / numeric helpers
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Lowercase + whitespace-collapsed key for local alignment fallbacks."""
    return " ".join((text or "").lower().split())


def _canon_entity(canonical_id: str, name: str) -> str:
    """Alignment key for an entity: ``canonical_id`` if set, else normed name."""
    cid = _norm(canonical_id)
    return cid or _norm(name)


def _first_number(text: str) -> float | None:
    """Extract the first numeric token from ``text`` (commas stripped)."""
    m = _NUM_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def values_match(a: str, b: str) -> bool:
    """Numeric-aware fuzzy equality of two asserted values.

    - If BOTH strings carry a number, they match when the numbers are within
      ±15% of each other (divide-by-zero guarded).
    - Otherwise a case-insensitive containment check: match when one normalized
      string contains the other (and both are non-empty).
    - Empty vs non-empty never matches.
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False

    fa, fb = _first_number(a), _first_number(b)
    if fa is not None and fb is not None:
        denom = max(abs(fa), abs(fb))
        if denom == 0:
            return fa == fb
        return abs(fa - fb) / denom <= VALUE_TOLERANCE

    return na in nb or nb in na


def _most_common(values: list[str]) -> str:
    """The modal non-empty string in ``values`` (first-seen wins ties)."""
    counts: Counter[str] = Counter(v for v in values if v)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


# Minimum surface-form length to text-verify on, so 1-2 char fragments ("fl",
# "us") can't false-positive against arbitrary page text.
_MIN_SURFACE_CHARS = 3


def _surface_matcher(names: list[str]) -> "re.Pattern | None":
    """A whole-word/phrase regex that matches ANY of an entity's surface forms.

    Used to verify an entity's PRESENCE against a page's raw text. The LLM is the
    *discoverer* of which entities matter (the brand union); whether a given page
    actually mentions one is a deterministic question we answer here, instead of
    trusting that page's isolated LLM call to have recalled it. Returns ``None``
    when there is no usable surface form (matcher is then a no-op).
    """
    forms: set[str] = set()
    for n in names or []:
        s = (n or "").strip().lower()
        if len(s) >= _MIN_SURFACE_CHARS:
            forms.add(re.escape(s))
    if not forms:
        return None
    # Longest first so the alternation prefers the most specific surface form.
    alt = "|".join(sorted(forms, key=len, reverse=True))
    try:
        return re.compile(r"\b(?:" + alt + r")\b")
    except re.error:
        return None


# --------------------------------------------------------------------------- #
# Alignment layer 1 — entity alias merge (brand-prefix / identical-core)
# --------------------------------------------------------------------------- #
# Query modifiers stripped from the keyword before topic matching.
_KW_MODIFIERS = {
    "best", "top", "cheap", "cheapest", "affordable", "good", "great",
    "near", "me", "review", "reviews", "buy", "vs", "versus", "how", "what",
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "with",
}

# Base stopwords that never make a token distinctive for core comparison.
_BASE_GENERIC = {
    "the", "a", "an", "and", "or", "of", "for", "with", "in", "on", "at", "to",
}


def _stem(tok: str) -> str:
    """Very light suffix strip so 'trainer'/'training'/'trainers' compare equal."""
    for suf in ("ings", "ing", "ers", "er", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: len(tok) - len(suf)]
    return tok


def _type_class(t: str) -> str:
    """Coarse class of an LLM free-form entity type string."""
    t = (t or "").lower()
    if any(k in t for k in ("person", "people", "trainer", "coach", "author",
                            "doctor", "founder", "instructor")):
        return "person"
    if any(k in t for k in ("location", "place", "city", "suburb", "region",
                            "country", "area", "state", "neighborhood",
                            "neighbourhood")):
        return "loc"
    if any(k in t for k in ("company", "organization", "organisation", "brand",
                            "business", "manufacturer", "retailer", "gym",
                            "studio", "clinic", "agency", "provider", "school",
                            "college", "university", "platform", "marketplace",
                            "facility", "centre", "center")):
        return "org"
    if any(k in t for k in ("product", "model", "item", "device", "tool",
                            "equipment")):
        return "product"
    if any(k in t for k in ("service", "procedure", "treatment", "program",
                            "class", "activity")):
        return "service"
    return "other"


def _keyword_stems(keyword: str) -> set[str]:
    return {
        _stem(t)
        for t in _norm(keyword).split()
        if t not in _KW_MODIFIERS
    }


def _build_alias_map(union_raw: dict[str, dict], keyword: str) -> dict[str, str]:
    """Map near-duplicate entity canons onto one winner canon.

    Two canons merge when their *distinctive cores* are identical: tokens minus
    brand tokens (from org-class entities), minus the keyword's own category
    tokens, minus base stopwords — stem-compared. So "Aeron Chair",
    "Herman Miller Aeron" and "Herman Miller Aeron Chair" all reduce to
    ``{aeron}`` and merge, while "Branch Ergonomic Chair" (core empty) and
    "Branch Ergonomic Chair Pro" (core ``{pro}``) stay apart, and any canon
    carrying a digit token ("Series 1", "Ignition 2.0") is never merged —
    digits mark model variants. Person/location entities never merge.

    The winner is the canon seen on the most pages (tie → longest, most
    specific). Returns ``{loser_canon: winner_canon}``; empty dict on any doubt.
    """
    kw_stems = _keyword_stems(keyword)

    cls: dict[str, str] = {}
    for c, meta in union_raw.items():
        cls[c] = _type_class(_most_common([t for t in meta.get("types", []) if t]))

    brand_toks: set[str] = set()
    for c in union_raw:
        if cls[c] == "org":
            brand_toks.update(c.split())

    def _core(c: str) -> frozenset | None:
        toks: list[str] = []
        for t in c.split():
            if t.isdigit():
                return None  # digit token = model variant, never merge
            st = _stem(t)
            if st in kw_stems or t in _BASE_GENERIC or t in brand_toks:
                continue
            if len(t) <= 2:
                return None  # cryptic short token ("pt", "v2") — don't risk it
            toks.append(st)
        return frozenset(toks) or None

    groups: dict[frozenset, list[str]] = {}
    for c in union_raw:
        if cls[c] in ("person", "loc"):
            continue
        core = _core(c)
        if core:
            groups.setdefault(core, []).append(c)

    alias: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(
            key=lambda c: (len(union_raw[c].get("pages", set())), len(c)),
            reverse=True,
        )
        win = members[0]
        for m in members[1:]:
            alias[m] = win
    return alias


# --------------------------------------------------------------------------- #
# Alignment layer 2 — first-party subject detection
# --------------------------------------------------------------------------- #
def _sld(domain: str) -> str:
    """The registrable label of a domain: 'www.chriswilsonpt.com' → 'chriswilsonpt'."""
    d = (domain or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]
    return d.split(".", 1)[0] if d else ""


def _brand_matches_domain(canon: str, sld: str) -> bool:
    """Does an entity canon look like the site's own brand name?

    Three signals: concatenated-name containment ("roar active" ↔ roar.com.au),
    initials prefix ("exclusive personal training" ↔ eptcoaching.com), or ≥2
    long name tokens embedded in the label ("chris wilson fitness studio" ↔
    chriswilsonpt.com).
    """
    if not canon or not sld:
        return False
    concat = canon.replace(" ", "")
    if len(concat) >= 4 and (concat in sld or sld in concat):
        return True
    toks = canon.split()
    initials = "".join(t[0] for t in toks if t)
    if len(initials) >= 3 and sld.startswith(initials):
        return True
    long_toks = [t for t in toks if len(t) >= 4]
    if len(long_toks) >= 2 and sum(1 for t in long_toks if t in sld) >= 2:
        return True
    return False


def _topic_canons(canons: set[str], keyword: str) -> set[str]:
    """Entity canons that ARE the keyword topic (stem-compared).

    A canon is the topic when its stem set equals the keyword's (modifiers
    stripped), or is a ≥2-stem subset of it ("personal training" ⊆
    "personal trainer perth"). Single-stem subsets stay out so bare locations
    ("perth") never become the topic.
    """
    kw = _keyword_stems(keyword)
    if not kw:
        return set()
    out: set[str] = set()
    for c in canons:
        stems = {_stem(t) for t in c.split()}
        if not stems:
            continue
        if stems == kw or (stems <= kw and len(stems) >= 2):
            out.add(c)
    return out


# Attribute families whose presence marks "this page states its own contact /
# business facts" — used to detect a page's self entity.
_CONTACT_FAMILIES = {"address", "phone", "email", "hours"}


def _provider_canons(
    page: PageEntities,
    alias: dict[str, str],
    page_counts: dict[str, int],
    n_brand: int,
) -> set[str]:
    """The page's OWN business entity canons (alias-resolved).

    An entity is the page's first party when its name matches the page's domain
    label, or when it carries ≥2 contact-family facts on this page while being
    (near-)unique to it across the SERP. Locations are never a first party.
    """
    out: set[str] = set()
    sld = _sld(getattr(page, "domain", "") or "")

    contact: Counter[str] = Counter()
    for t in getattr(page, "triples", []) or []:
        if getattr(t, "is_edge", False):
            continue
        c0 = _canon_entity(getattr(t, "canonical_entity", ""), getattr(t, "entity", ""))
        c = alias.get(c0, c0)
        if _attr_family(_norm(getattr(t, "attribute", "") or "")) in _CONTACT_FAMILIES:
            contact[c] += 1

    uniq_cap = max(1, round(0.15 * max(1, n_brand)))
    for e in getattr(page, "entities", []) or []:
        c0 = _canon_entity(getattr(e, "canonical_id", ""), getattr(e, "name", ""))
        c = alias.get(c0, c0)
        if not c or _type_class(getattr(e, "type", "")) == "loc":
            continue
        if sld and _brand_matches_domain(c, sld):
            out.add(c)
            continue
        if contact.get(c, 0) >= 2 and page_counts.get(c, 0) <= uniq_cap:
            out.add(c)
    return out


# --------------------------------------------------------------------------- #
# Alignment layer 3 — deterministic attribute families
# --------------------------------------------------------------------------- #
# Fallback collapse for obvious attribute families. The LLM corpus
# canonicalizer (clients/entities.py) does the nuanced work when it succeeds;
# this keeps the table usable when it flaps (observed live 2026-07-11:
# member_11_30min_rate / low_cost / median_cost all survived as separate rows).
_ATTR_FAMILY_PATTERNS: list[tuple[str, "re.Pattern"]] = [
    ("rating", re.compile(r"rating|review_score|stars")),
    ("price", re.compile(r"price|cost|rate\b|_rate|fee|pricing|charge")),
    ("address", re.compile(r"address|street_|location$")),
    ("phone", re.compile(r"phone|telephone|mobile|contact_number")),
    ("email", re.compile(r"email|e_mail")),
    ("hours", re.compile(r"hours|opening_time|session_time|schedule")),
    ("warranty", re.compile(r"warranty|guarantee")),
    ("experience", re.compile(r"experience|years_in|established|founded|qualified_since")),
]


def _attr_family(norm_attr: str) -> str:
    """Collapse a normalized attribute name onto its family, or return it as-is."""
    a = norm_attr or ""
    for fam, rx in _ATTR_FAMILY_PATTERNS:
        if rx.search(a):
            return fam
    return a


# First-party attribute families whose values are quantities worth ranging.
# Address/phone/email digits must never be ranged ("12–502" is not an address
# consensus) — those get the "each page states its own" treatment instead.
_RANGE_FAMILIES = {"price", "rating", "experience", "warranty"}


def _range_display(values: list[str]) -> str:
    """Consensus display for first-party numeric values: '$40–$94' style range.

    First-party values (each page's own price/experience) are inherently
    different per business, so a modal value is meaningless — show the observed
    range instead. Falls back to '' when fewer than two values parse numeric.
    """
    nums: list[float] = []
    has_currency = False
    for v in values:
        n = _first_number(v)
        if n is None:
            continue
        nums.append(n)
        if "$" in v or "€" in v or "£" in v:
            has_currency = True
    if len(nums) < 2:
        return ""
    lo, hi = min(nums), max(nums)
    # Heterogeneous magnitudes ("19 years" vs "founded 1999") make a nonsense
    # range — punt to the identity treatment instead of printing "19–1,999".
    if lo > 0 and hi / lo > 50:
        return ""

    def _fmt(x: float) -> str:
        s = f"{x:,.2f}".rstrip("0").rstrip(".")
        return f"${s}" if has_currency else s

    if lo == hi:
        return _fmt(lo)
    return f"{_fmt(lo)}–{_fmt(hi)}"


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def build_entity_table(
    pages: list[PageEntities],
    target: PageEntities | None,
    keyword: str,
    top_n: int = 20,
) -> EntityTable:
    """Build the top-N topical EAV comparison table for a keyword.

    Args:
        pages: the top-N ranking pages (the "brands"), in any order. Consensus is
            derived from these only.
        target: the tracked target page (rendered as the ``rank == 0`` column),
            or ``None`` when no URL is tracked.
        keyword: the analysis keyword (echoed onto the table + summary).
        top_n: cap on how many brand pages contribute columns / consensus.

    Returns:
        A populated :class:`EntityTable` — two matrices (View A coverage, View B
        EAV claims) plus the gap summary. Returns an empty-but-valid table on
        degenerate input; never raises.
    """
    try:
        return _build(pages, target, keyword, top_n)
    except Exception:  # noqa: BLE001 — entity table is opt-in context, never fatal
        return EntityTable(keyword=keyword, has_target=target is not None)


def _build(
    pages: list[PageEntities],
    target: PageEntities | None,
    keyword: str,
    top_n: int,
) -> EntityTable:
    # 1) Brand columns: sort by rank asc, cap to top_n. ---------------------- #
    brands = sorted(
        [p for p in (pages or []) if p is not None],
        key=lambda p: getattr(p, "rank", 0),
    )[: max(0, top_n)]

    # Columns in display order: target (rank 0) first, then brands by rank.
    ranks: list[int] = []
    rank_domains: dict[int, str] = {}
    if target is not None:
        ranks.append(0)
        rank_domains[0] = getattr(target, "domain", "") or ""
    for b in brands:
        r = getattr(b, "rank", 0)
        ranks.append(r)
        rank_domains[r] = getattr(b, "domain", "") or ""

    if not brands and target is None:
        return EntityTable(keyword=keyword, has_target=False, ranks=ranks,
                           rank_domains=rank_domains)

    all_pages: list[PageEntities] = brands + ([target] if target is not None else [])

    # Did the target's entity extraction actually read the page? A page with
    # real body text but zero entities AND zero triples is an LLM flap, not an
    # empty page — grading attribute gaps against it would be confidently
    # wrong. View A presence stays valid either way (it is text-verified).
    target_read_ok = True
    if target is not None:
        has_body = len(getattr(target, "body_text", "") or "") >= 300
        if has_body and not (getattr(target, "entities", None) or []) \
                and not (getattr(target, "triples", None) or []):
            target_read_ok = False

    # 1b) Alignment pre-pass: raw union → alias map, topic + per-page subject. #
    union_probe: dict[str, dict] = {}
    for p in all_pages:
        for e in getattr(p, "entities", []) or []:
            canon = _canon_entity(getattr(e, "canonical_id", ""), getattr(e, "name", ""))
            if not canon:
                continue
            agg = union_probe.setdefault(canon, {"names": [], "types": [], "pages": set()})
            agg["names"].append(getattr(e, "name", "") or "")
            agg["types"].append(getattr(e, "type", "") or "")
            agg["pages"].add(id(p))

    alias = _build_alias_map(union_probe, keyword)

    def A(canon: str) -> str:
        return alias.get(canon, canon)

    # Brand-page counts per alias-resolved canon (for provider uniqueness).
    page_counts: Counter[str] = Counter()
    for b in brands:
        seen: set[str] = set()
        for e in getattr(b, "entities", []) or []:
            c = A(_canon_entity(getattr(e, "canonical_id", ""), getattr(e, "name", "")))
            if c and c not in seen:
                seen.add(c)
                page_counts[c] += 1

    topic = {A(c) for c in _topic_canons(set(union_probe.keys()), keyword)}
    n_brand = len(brands)
    subjects_by_page: dict[int, set[str]] = {}
    for p in all_pages:
        subjects_by_page[id(p)] = topic | _provider_canons(p, alias, page_counts, n_brand)

    # 2) Per-page entity index: canon -> {salience, name, type}. ------------- #
    # Build an index for every page (brands + target) keyed by canon entity id.
    def _entity_index(page: PageEntities) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        for e in getattr(page, "entities", []) or []:
            canon = A(_canon_entity(getattr(e, "canonical_id", ""), getattr(e, "name", "")))
            if not canon:
                continue
            sal = getattr(e, "salience", 0.0) or 0.0
            prev = idx.get(canon)
            if prev is None or sal > prev["salience"]:
                idx[canon] = {
                    "salience": sal,
                    "name": getattr(e, "name", "") or canon,
                    "type": getattr(e, "type", "") or "",
                    "names": prev["names"] if prev else [],
                    "types": prev["types"] if prev else [],
                }
            # Track all surface forms / types for modal display selection.
            idx[canon].setdefault("names", []).append(getattr(e, "name", "") or "")
            idx[canon].setdefault("types", []).append(getattr(e, "type", "") or "")
        return idx

    brand_eidx = [(b, _entity_index(b)) for b in brands]
    target_eidx = _entity_index(target) if target is not None else {}

    # Per-page lowercased body text, for deterministic presence verification (View
    # A). Falls back to "" when a page carries no body — the matcher is then a
    # no-op for that column, so behaviour is unchanged for text-less inputs.
    brand_text: dict[int, str] = {
        getattr(b, "rank", 0): (getattr(b, "body_text", "") or "").lower()
        for b in brands
    }
    target_text = (getattr(target, "body_text", "") or "").lower() if target is not None else ""

    # 3) Per-page triple index: (canon, family_attr) -> {value, attr, is_edge}. #
    # Subject remap: this page's own business + the keyword topic land on the
    # shared __subject__ row so first-party claims align across pages.
    def _triple_index(page: PageEntities) -> dict[tuple[str, str], dict]:
        subjects = subjects_by_page.get(id(page), set())
        idx: dict[tuple[str, str], dict] = {}
        for t in getattr(page, "triples", []) or []:
            canon = A(_canon_entity(
                getattr(t, "canonical_entity", ""), getattr(t, "entity", "")
            ))
            attr = getattr(t, "attribute", "") or ""
            if not canon or not _norm(attr):
                continue
            is_edge = bool(getattr(t, "is_edge", False))
            if canon in subjects:
                canon = SUBJECT_KEY
            norm_attr = _norm(attr)
            fam = norm_attr if is_edge else _attr_family(norm_attr)
            key = (canon, fam)
            val = getattr(t, "value", "") or ""
            prev = idx.get(key)
            # First non-empty value wins; keep the display attribute / edge flag.
            if prev is None:
                idx[key] = {
                    "value": val,
                    "attr": fam if fam != norm_attr else attr,
                    "is_edge": is_edge,
                }
            elif not prev["value"] and val:
                prev["value"] = val
        return idx

    brand_tidx = [(b, _triple_index(b)) for b in brands]
    target_tidx = _triple_index(target) if target is not None else {}

    # ---------------------------------------------------------------------- #
    # View A — coverage_rows (one row per entity, over the BRAND union).
    # ---------------------------------------------------------------------- #
    # Union of canon entities across brands, with aggregated display metadata.
    union_entities: dict[str, dict] = {}
    for _b, idx in brand_eidx:
        for canon, meta in idx.items():
            agg = union_entities.setdefault(
                canon, {"names": [], "types": [], "salience": 0.0}
            )
            agg["names"].extend(meta.get("names", []))
            agg["types"].extend(meta.get("types", []))
            agg["salience"] = max(agg["salience"], meta.get("salience", 0.0))

    brand_slds = {
        _sld(rank_domains.get(r, "")) for r in ranks if r != 0
    } - {""}

    coverage_rows: list[EavRow] = []
    coverage_canons: dict[int, str] = {}  # row index -> canon (for must_add guard)
    for canon, agg in union_entities.items():
        display = _most_common([n for n in agg["names"] if n]) or canon
        etype = _most_common([t for t in agg["types"] if t])

        # Whole-word matcher over every surface form seen for this entity, plus
        # its display name. Presence is (LLM recalled it) OR (the page text
        # actually contains it) — monotonic, so it only ever ADDS a true hit a
        # page's isolated LLM call missed; it never hides one.
        matcher = _surface_matcher(list(agg["names"]) + [display, canon])

        cells: dict[int, EntityCell] = {}
        consensus = 0
        for b, idx in brand_eidx:
            r = getattr(b, "rank", 0)
            hit = idx.get(canon)
            present = hit is not None
            if not present and matcher is not None and brand_text.get(r):
                present = matcher.search(brand_text[r]) is not None
            if present:
                consensus += 1
            cells[r] = EntityCell(
                rank=r,
                present=present,
                salience=round(hit["salience"], 4) if hit else 0.0,
            )

        target_present = False
        if target is not None:
            t_hit = target_eidx.get(canon)
            target_present = t_hit is not None
            if not target_present and matcher is not None and target_text:
                target_present = matcher.search(target_text) is not None
            cells[0] = EntityCell(
                rank=0,
                present=target_present,
                salience=round(t_hit["salience"], 4) if t_hit else 0.0,
            )

        coverage_canons[len(coverage_rows)] = canon
        coverage_rows.append(
            EavRow(
                entity=display,
                entity_type=etype,
                attribute="",
                is_edge=False,
                cells=cells,
                consensus=consensus,
                graded=consensus >= MIN_GAP_CONSENSUS,
                target_present=target_present,
                gap="" if (target is None or target_present) else "missing_entity",
            )
        )

    # Sort by consensus desc, then by best brand salience desc (keep the canon
    # mapping aligned through the sort for the must_add guard below).
    order = sorted(
        range(len(coverage_rows)),
        key=lambda i: (
            coverage_rows[i].consensus,
            max((c.salience for c in coverage_rows[i].cells.values()), default=0.0),
        ),
        reverse=True,
    )
    coverage_rows = [coverage_rows[i] for i in order][:MAX_COVERAGE_ROWS]
    row_canons = [coverage_canons[i] for i in order][:MAX_COVERAGE_ROWS]

    # ---------------------------------------------------------------------- #
    # View B — eav_rows (one row per (entity, attribute) over the BRAND union).
    # ---------------------------------------------------------------------- #
    union_pairs: dict[tuple[str, str], dict] = {}
    for _b, idx in brand_tidx.copy():
        for key, meta in idx.items():
            agg = union_pairs.setdefault(
                key, {"attrs": [], "is_edge": False}
            )
            agg["attrs"].append(meta["attr"])
            agg["is_edge"] = agg["is_edge"] or meta["is_edge"]

    # Which canon entities does the target cover at all? (for gap classification)
    target_entity_canons = set(target_eidx.keys()) if target is not None else set()
    # The target always "covers" the subject row — it IS its own first party.
    target_entity_canons.add(SUBJECT_KEY)

    # Family fallback for subject rows: everything on the target page is
    # first-party context, so if it states a price/experience/… on ANY of its
    # own entities ("Personalised 1:1 Coaching" → price), the page DOES state
    # one — don't tell the owner to add a fact that's already there just
    # because the LLM hung it off a sub-entity.
    target_any_by_fam: dict[str, str] = {}
    for (_c, fam), meta in (target_tidx or {}).items():
        if meta.get("is_edge"):
            continue
        v = meta.get("value") or ""
        if v and fam not in target_any_by_fam:
            target_any_by_fam[fam] = v

    eav_rows: list[EavRow] = []
    edge_rows: list[EavRow] = []
    for (canon, norm_attr), agg in union_pairs.items():
        display_attr = _most_common([a for a in agg["attrs"] if a]) or norm_attr
        is_edge = agg["is_edge"]
        is_subject = canon == SUBJECT_KEY

        # Entity display name: reuse the brand entity union if known.
        if is_subject:
            entity_disp = SUBJECT_DISPLAY
            entity_type = SUBJECT_TYPE
        else:
            ent_meta = union_entities.get(canon)
            if ent_meta:
                entity_disp = _most_common([n for n in ent_meta["names"] if n]) or canon
                entity_type = _most_common([t for t in ent_meta["types"] if t])
            else:
                entity_disp = canon
                entity_type = ""

        cells: dict[int, EntityCell] = {}
        consensus = 0
        brand_values: list[tuple[int, str]] = []  # (rank, value) for modal pick
        for b, idx in brand_tidx:
            r = getattr(b, "rank", 0)
            hit = idx.get((canon, norm_attr))
            present = hit is not None
            val = hit["value"] if hit else ""
            if present:
                consensus += 1
                if val:
                    brand_values.append((r, val))
            cells[r] = EntityCell(rank=r, present=present, value=val)

        # Consensus display: first-party values are inherently per-business —
        # quantitative families show the observed range ('$40–$94'), identity
        # families ('each page states its own') never pretend one page's phone
        # number is a consensus. Everything else: modal value (tie -> best rank).
        consensus_value = ""
        if is_subject and not is_edge:
            if norm_attr in _RANGE_FAMILIES:
                consensus_value = _range_display([v for _r, v in brand_values])
            if not consensus_value and len({_norm(v) for _r, v in brand_values if v}) >= 2:
                consensus_value = "each page states its own"
        if not consensus_value:
            consensus_value = _modal_value(brand_values)
        modal_backing = sum(
            1 for _r, v in brand_values if v and values_match(v, consensus_value)
        )

        # Target column + consensus-gated gap classification. Contact facts
        # (address/phone/email/hours) about a NON-subject entity are never your
        # gap — directories cross-list the same local businesses, so a rival's
        # address can reach consensus=2, and "state Extension Fitness's address"
        # is not advice. Your own contact facts live on the subject row.
        third_party_contact = (
            not is_subject and not is_edge and norm_attr in _CONTACT_FAMILIES
        )
        graded = (
            target is not None
            and target_read_ok
            and consensus >= MIN_GAP_CONSENSUS
            and not third_party_contact
        )
        target_present = False
        target_value = ""
        gap = ""
        if target is not None:
            t_hit = target_tidx.get((canon, norm_attr))
            target_present = t_hit is not None
            target_value = t_hit["value"] if t_hit else ""
            if is_subject and not target_present:
                fb = target_any_by_fam.get(norm_attr, "")
                if fb:
                    target_present = True
                    target_value = fb
            cells[0] = EntityCell(
                rank=0, present=target_present, value=target_value
            )
            if graded:
                gap = _classify_gap(
                    target_present=target_present,
                    target_value=target_value,
                    consensus_value=consensus_value,
                    entity_on_target=canon in target_entity_canons,
                )
                # "Off consensus" needs a consensus to be off: first-party
                # values differ per business by nature, and a modal value only
                # one page asserts is not a consensus.
                if gap == "off_consensus" and (is_subject or modal_backing < 2):
                    gap = ""

        row = EavRow(
            entity=entity_disp,
            entity_type=entity_type,
            attribute=display_attr,
            is_edge=is_edge,
            role="subject" if is_subject else "",
            cells=cells,
            consensus=consensus,
            graded=graded,
            consensus_value=consensus_value,
            target_present=target_present,
            target_value=target_value,
            gap=gap,
        )
        # Attribute->value claims and entity->entity relationships are shown in
        # two distinct matrices; route the row to the right one.
        (edge_rows if is_edge else eav_rows).append(row)

    # Sort each: graded gap rows first, then graded-covered, then page-specific
    # (ungraded) intel — consensus desc within each band.
    _row_sort = lambda row: (  # noqa: E731
        0 if (row.graded and row.gap) else (1 if row.graded else 2),
        -row.consensus,
    )
    eav_rows.sort(key=_row_sort)
    edge_rows.sort(key=_row_sort)
    eav_rows = eav_rows[:MAX_EAV_ROWS]
    edge_rows = edge_rows[:MAX_EAV_ROWS]

    # ---------------------------------------------------------------------- #
    # Table-level fields.
    # ---------------------------------------------------------------------- #
    n_entities = len(union_entities)
    # Split union pairs into attribute->value claims vs entity->entity edges.
    edge_keys = {key for key, agg in union_pairs.items() if agg.get("is_edge")}
    attr_keys = {key for key in union_pairs if key not in edge_keys}
    n_pairs = len(attr_keys)
    n_edges = len(edge_keys)
    n_graded_pairs = sum(1 for r in eav_rows if r.graded)
    n_graded_edges = sum(1 for r in edge_rows if r.graded)

    # must_add: high-consensus entities the target is entirely missing.
    # Competitor self-brands are excluded — "add your rival's brand name to
    # your page" is not advice (their name ↔ their ranking domain).
    n_brand_with_entities = sum(1 for _b, idx in brand_eidx if idx)
    add_threshold = max(3, round(0.5 * n_brand_with_entities))
    must_add_entities: list[str] = []
    if target is not None:
        for row, canon in zip(coverage_rows, row_canons):
            if row.gap != "missing_entity" or row.consensus < add_threshold:
                continue
            if any(_brand_matches_domain(canon, sld) for sld in brand_slds):
                continue
            must_add_entities.append(row.entity)
    must_add_entities = must_add_entities[:MAX_MUST_ADD]

    # target_completeness: graded attribute claims the target covers / graded
    # total (edges excluded). Page-specific (consensus=1) claims don't count —
    # nobody should be graded against one competitor's private facts.
    target_completeness = None
    covered = 0
    if target is not None and n_graded_pairs:
        covered = sum(
            1
            for r in eav_rows
            if r.graded
            and r.gap not in ("missing_entity", "missing_attribute", "missing_value")
        )
        target_completeness = round(covered / n_graded_pairs, 3)

    # Summary line. -------------------------------------------------------- #
    k = n_brand or len(ranks)
    if target is not None and not target_read_ok:
        summary = (
            f"{n_entities} topical entities across the top {k}. "
            "⚠ Your page's fact extraction failed this run, so attribute-level "
            "gaps are not graded — entity coverage (text-verified) is still valid. "
            "Re-run the analysis for the full comparison."
        )
    elif target is not None:
        summary = (
            f"{n_entities} topical entities across the top {k}; "
            f"{n_graded_pairs} consensus attribute claims (asserted by 2+ pages) — "
            f"your page covers {covered}/{n_graded_pairs}; "
            f"{len(must_add_entities)} entities missing entirely. "
            f"({n_pairs - n_graded_pairs} page-specific claims kept as market intel.)"
        )
    else:
        summary = (
            f"{n_entities} topical entities and {n_pairs} attribute claims "
            f"({n_graded_pairs} consensus) across the top {k} — no target page tracked."
        )

    return EntityTable(
        keyword=keyword,
        ranks=ranks,
        rank_domains=rank_domains,
        has_target=target is not None,
        coverage_rows=coverage_rows,
        eav_rows=eav_rows,
        edge_rows=edge_rows,
        must_add_entities=must_add_entities,
        target_completeness=target_completeness,
        target_read_ok=target_read_ok,
        n_entities=n_entities,
        n_pairs=n_pairs,
        n_edges=n_edges,
        n_graded_pairs=n_graded_pairs,
        n_graded_edges=n_graded_edges,
        summary=summary,
        cost_usd=0.0,
    )


# --------------------------------------------------------------------------- #
# Gap / consensus helpers
# --------------------------------------------------------------------------- #
def _modal_value(rank_values: list[tuple[int, str]]) -> str:
    """Modal non-empty value across brand pages; ties -> best (lowest) rank.

    ``rank_values`` is ``[(rank, value), ...]``. The most frequent value wins; on
    a frequency tie the value coming from the lowest rank (best SERP position) is
    chosen.
    """
    if not rank_values:
        return ""
    counts: Counter[str] = Counter(v for _r, v in rank_values if v)
    if not counts:
        return ""
    top = counts.most_common()
    best_freq = top[0][1]
    tied = [val for val, freq in top if freq == best_freq]
    if len(tied) == 1:
        return tied[0]
    # Tie-break: the tied value asserted by the lowest-rank page.
    best_rank = None
    best_val = tied[0]
    for r, v in rank_values:
        if v in tied and (best_rank is None or r < best_rank):
            best_rank = r
            best_val = v
    return best_val


def _classify_gap(
    target_present: bool,
    target_value: str,
    consensus_value: str,
    entity_on_target: bool,
) -> str:
    """Classify the target's gap for one ``(entity, attribute)`` pair.

    Order of checks matters:
      - target asserts a matching value           -> ``""``
      - target asserts a non-matching value        -> ``"off_consensus"``
      - the entity is entirely absent from target  -> ``"missing_entity"``
      - the entity is present but this attr absent  -> ``"missing_attribute"``
      - the attr row exists but with empty value    -> ``"missing_value"``
    """
    if target_present and target_value:
        if values_match(target_value, consensus_value):
            return ""
        return "off_consensus"
    # No usable value from the target for this pair.
    if not entity_on_target:
        return "missing_entity"
    if not target_present:
        return "missing_attribute"
    # Row exists on target but value is empty.
    return "missing_value"
