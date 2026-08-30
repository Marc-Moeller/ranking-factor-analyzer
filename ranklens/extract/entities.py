"""Entity / EAV extraction — merge the LLM read with author-declared schema.

This module turns a page's raw LLM entity output plus its JSON-LD nodes into one
``PageEntities`` (the contract in ``ranklens.models``), and derives the local
entity *factor* values (title/heading/body location counts) with NO network and
NO LLM calls — those are pure functions of the already-fetched HTML + body text.

Schema-declared truth wins: on an ``(entity, attribute)`` conflict between an LLM
triple and a JSON-LD triple, the author-declared schema triple is kept.

Robustness is a hard requirement: one malformed page must never crash a run.
Every public function wraps its risky parts and degrades to empty/0.0.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ..keywords import split_sentences
from ..models import EavTriple, EntityMention, PageEntities

# tags whose text never belongs in the visible title/heading surface
_STRIP_TAGS = ("script", "style", "noscript", "template", "svg")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Document/CMS/navigation chrome that an LLM may still leak as "entities".
# Compared against canon(name); kept tight so real entities survive.
_NAV_ENTITY_STOP = {
    "home", "menu", "search", "next", "previous", "back", "article",
    "blog", "blog post", "category", "tag", "page", "web page", "webpage",
    "website", "breadcrumb", "navigation", "footer", "header", "sidebar",
    "comment", "login", "sign in", "cart", "checkout", "account", "faq",
    "buying guide", "table of content", "newsletter", "subscribe", "share",
}

# Document/SEO metadata attributes that are NEVER real-world EAV facts.
# Compared against the normalized snake_case attribute name.
_META_ATTR_STOP = {
    "datepublished", "datemodified", "datecreated", "uploaddate",
    "inlanguage", "language", "url", "headline", "wordcount", "articlesection",
    "position", "item", "nextitem", "previousitem", "previtem", "identifier",
    "mainentityofpage", "ispartof", "potentialaction", "thumbnailurl",
    "thumbnail", "contenturl", "embedurl", "commentcount", "articlebody",
    "publisher", "author", "slug", "permalink", "canonical", "robots",
    "image", "logo", "sameas",
}

# Small synonym alias map to collapse near-duplicate attribute names across the
# LLM and schema reads. Kept intentionally tiny — only obvious 1:1 collapses.
_ATTR_ALIASES = {
    "cost": "price",
    "pricing": "price",
    "delivery time": "delivery",
    "shipping": "delivery",
    "phone number": "telephone",
    "opening hours": "hours",
    "hours of operation": "hours",
}


def canon(name: str) -> str:
    """Canonical alignment key: lowercase, strip punctuation, collapse whitespace,
    singularize a trailing 's' on the last token. Stable across pages."""
    try:
        if not name:
            return ""
        s = _PUNCT_RE.sub(" ", str(name).lower())
        s = _WS_RE.sub(" ", s).strip()
        if not s:
            return ""
        toks = s.split(" ")
        last = toks[-1]
        # naive singularization: drop a trailing plural 's' (but keep 'ss', 'is')
        if len(last) > 3 and last.endswith("s") and not last.endswith("ss") and not last.endswith("is"):
            last = last[:-1]
            toks[-1] = last
        return " ".join(toks)
    except Exception:
        return ""


def _norm_attr(attribute: str) -> str:
    """Normalize an attribute name: lowercase, snake_case, collapse whitespace,
    then apply the small synonym alias map."""
    try:
        if not attribute:
            return ""
        a = str(attribute).strip().lower()
        a = _WS_RE.sub(" ", a)
        # apply the alias map on the human-readable form first (handles spaces)
        a = _ATTR_ALIASES.get(a, a)
        # snake_case: spaces/hyphens -> underscore, drop other punctuation
        a = a.replace("-", " ").replace("/", " ")
        a = _WS_RE.sub("_", a.strip())
        a = re.sub(r"[^\w]", "", a)
        # alias map again on the snake form (e.g. if it arrived pre-snaked)
        a = _ATTR_ALIASES.get(a, a)
        return a
    except Exception:
        return ""


def _soup(html: str):
    """BeautifulSoup with lxml + html.parser fallback (mirrors factors.py)."""
    html = html or ""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        try:
            return BeautifulSoup(html, "html.parser")
        except Exception:
            return None


def build_page_entities(
    raw_llm: dict,
    html: str,
    body_text: str,
    url: str,
    rank: int,
    domain: str,
) -> "PageEntities":
    """Build one PageEntities from the page's LLM knowledge-graph read.

    EAV is LLM-only by design: author-declared JSON-LD is document/CMS plumbing
    (ListItem.position, Article.datePublished, breadcrumb nodes, ...) and pollutes
    the topical comparison, so it is NOT merged here. ``html``/``body_text`` are
    retained for the (separate) local entity-location factors.
    """
    raw_llm = raw_llm or {}

    # ---- entities (LLM) ------------------------------------------------------
    # keep one EntityMention per canon(name), keeping the highest salience
    ent_by_key: dict[str, EntityMention] = {}
    try:
        for e in raw_llm.get("entities") or []:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            key = canon(name)
            if not key or key in _NAV_ENTITY_STOP:
                continue  # drop navigation / document chrome leaked as an entity
            try:
                sal = float(e.get("salience") or 0.0)
            except Exception:
                sal = 0.0
            etype = str(e.get("type") or "").strip()
            prev = ent_by_key.get(key)
            if prev is None or sal > prev.salience:
                ent_by_key[key] = EntityMention(
                    name=name,
                    type=etype,
                    salience=sal,
                    canonical_id=key,
                    source="llm",
                )
    except Exception:
        pass

    # ---- triples (LLM): plain triples + edges --------------------------------
    # keyed by (canon(entity), normalized attribute); schema can overwrite later
    triple_by_key: dict[tuple[str, str], EavTriple] = {}

    def _add_triple(entity: str, attribute: str, value: str, is_edge: bool, source: str) -> None:
        entity = str(entity or "").strip()
        value = str(value or "").strip()
        attr = _norm_attr(attribute)
        if not entity or not attr or not value:
            return
        # Drop document/SEO metadata claims (non-edges only — edge relations like
        # "publisher"/"author" are legitimate real-world connections).
        if not is_edge and attr in _META_ATTR_STOP:
            return
        if canon(entity) in _NAV_ENTITY_STOP:
            return  # claim about navigation/document chrome, not a real entity
        key = (canon(entity), attr)
        existing = triple_by_key.get(key)
        # schema wins over llm; otherwise first-writer keeps the slot
        if existing is not None and not (source == "schema" and existing.source != "schema"):
            return
        triple_by_key[key] = EavTriple(
            entity=entity,
            attribute=attr,
            value=value,
            canonical_entity=canon(entity),
            is_edge=is_edge,
            source=source,
        )

    try:
        for t in raw_llm.get("triples") or []:
            if not isinstance(t, dict):
                continue
            _add_triple(
                t.get("entity"),
                t.get("attribute"),
                t.get("value"),
                is_edge=False,
                source="llm",
            )
    except Exception:
        pass

    try:
        for ed in raw_llm.get("edges") or []:
            if not isinstance(ed, dict):
                continue
            # edge schema: relation -> attribute, target -> value
            _add_triple(
                ed.get("entity"),
                ed.get("relation"),
                ed.get("target"),
                is_edge=True,
                source="llm",
            )
    except Exception:
        pass

    return PageEntities(
        url=url,
        rank=rank,
        domain=domain,
        entities=list(ent_by_key.values()),
        triples=list(triple_by_key.values()),
        body_text=body_text or "",
    )


def apply_attribute_map(pe: "PageEntities", attr_map: dict) -> None:
    """Rewrite this page's attribute names onto the shared canonical vocabulary.

    ``attr_map`` is ``{raw_norm_attr: canonical_norm_attr}`` from the corpus-level
    canonicalizer. Mutates ``pe.triples`` in place. Only attribute->value facts
    are remapped; entity->entity edges (``is_edge``) keep their relation name.
    No-op on an empty map or a page with no triples. Never raises.
    """
    if not attr_map or pe is None:
        return
    try:
        for t in pe.triples or []:
            if getattr(t, "is_edge", False):
                continue
            canon = attr_map.get(t.attribute)
            if canon and canon != t.attribute:
                t.attribute = canon
    except Exception:
        return


def _title_text(soup) -> str:
    try:
        el = soup.find("title")
        return el.get_text(" ", strip=True) if el else ""
    except Exception:
        return ""


def _heading_text(soup, level: int) -> str:
    """Concatenated text of all <hN> at one level (mirrors factors.py)."""
    try:
        parts = []
        for h in soup.find_all(f"h{level}"):
            txt = h.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
        return " ".join(parts)
    except Exception:
        return ""


def _surface_forms(pe: "PageEntities") -> list[tuple[str, str]]:
    """(lower surface form, lower canonical form) for each entity, deduped."""
    forms: list[tuple[str, str]] = []
    seen: set[str] = set()
    for e in pe.entities:
        surf = (e.name or "").strip().lower()
        can = (e.canonical_id or canon(e.name)).strip().lower()
        if not surf and not can:
            continue
        key = e.canonical_id or canon(e.name)
        if key in seen:
            continue
        seen.add(key)
        forms.append((surf, can))
    return forms


def _appears(surf: str, can: str, target_low: str) -> bool:
    """Case-insensitive substring match of surface OR canonical form."""
    if not target_low:
        return False
    if surf and surf in target_low:
        return True
    if can and can in target_low:
        return True
    return False


def entity_factors(
    pe: "PageEntities",
    html: str,
    body_text: str,
) -> dict[str, float]:
    """Local location counts -> the entity factor ids (NO network, NO LLM)."""
    out: dict[str, float] = {
        "ENTITIES_TITLE": 0.0,
        "ENTITIES_H1": 0.0,
        "ENTITIES_H2": 0.0,
        "ENTITIES_H3": 0.0,
        "ENTITIES_BODY": 0.0,
        "ENTITIES_SENTENCES": 0.0,
        "DISTINCT_ENTITIES": 0.0,
        "ENTITY_SALIENCE_SUM": 0.0,
    }

    forms = _surface_forms(pe)
    soup = _soup(html)

    def _count_in(target_text: str) -> float:
        low = (target_text or "").lower()
        if not low:
            return 0.0
        return float(sum(1 for surf, can in forms if _appears(surf, can, low)))

    # ---- title / headings (parsed from html like factors.py) -----------------
    try:
        out["ENTITIES_TITLE"] = _count_in(_title_text(soup)) if soup is not None else 0.0
    except Exception:
        out["ENTITIES_TITLE"] = 0.0
    try:
        out["ENTITIES_H1"] = _count_in(_heading_text(soup, 1)) if soup is not None else 0.0
    except Exception:
        out["ENTITIES_H1"] = 0.0
    try:
        out["ENTITIES_H2"] = _count_in(_heading_text(soup, 2)) if soup is not None else 0.0
    except Exception:
        out["ENTITIES_H2"] = 0.0
    try:
        out["ENTITIES_H3"] = _count_in(_heading_text(soup, 3)) if soup is not None else 0.0
    except Exception:
        out["ENTITIES_H3"] = 0.0

    # ---- body ---------------------------------------------------------------
    try:
        out["ENTITIES_BODY"] = _count_in(body_text)
    except Exception:
        out["ENTITIES_BODY"] = 0.0

    # ---- per-sentence mentions, summed --------------------------------------
    try:
        total = 0
        for sent in split_sentences(body_text or ""):
            low = sent.lower()
            total += sum(1 for surf, can in forms if _appears(surf, can, low))
        out["ENTITIES_SENTENCES"] = float(total)
    except Exception:
        out["ENTITIES_SENTENCES"] = 0.0

    # ---- pure entity counters -----------------------------------------------
    try:
        out["DISTINCT_ENTITIES"] = float(len(pe.entities))
    except Exception:
        out["DISTINCT_ENTITIES"] = 0.0
    try:
        out["ENTITY_SALIENCE_SUM"] = float(sum(e.salience for e in pe.entities))
    except Exception:
        out["ENTITY_SALIENCE_SUM"] = 0.0

    return out
