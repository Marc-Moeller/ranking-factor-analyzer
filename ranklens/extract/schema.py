"""Structured-data factors — JSON-LD + microdata.

Parses every ``<script type="application/ld+json">`` block (tolerating malformed
JSON, top-level lists, and ``@graph`` wrappers) plus microdata
(``itemtype`` / ``itemprop``) and returns the Schema-group factor values keyed by
their ``factors_registry`` ids.

All ``@type`` matching is case-insensitive.
"""
from __future__ import annotations

import json
import re

# ---- canonical type buckets (lowercased) -------------------------------------
_ORG_LOCALBIZ = {"organization", "localbusiness"}
_PRODUCT_OFFER = {"product", "offer", "aggregateoffer"}
_AGG_RATING = {"aggregaterating"}
_FAQ = {"faqpage", "question"}
_BREADCRUMB = {"breadcrumblist"}

# property keys that imply a Product/Offer even without an explicit @type
_OFFER_PRICE_KEYS = {"price", "highprice", "lowprice", "pricerange", "offercount"}
_RATING_KEYS = {"ratingvalue", "reviewcount", "ratingcount"}


def _iter_jsonld_nodes(obj):
    """Yield every dict node in a JSON-LD object, unwrapping @graph and lists."""
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_jsonld_nodes(item)
        return
    if not isinstance(obj, dict):
        return
    yield obj
    # @graph holds an array of further nodes
    graph = obj.get("@graph")
    if graph is not None:
        yield from _iter_jsonld_nodes(graph)
    # nested objects can carry their own @type (e.g. offers, aggregateRating)
    for v in obj.values():
        if isinstance(v, (dict, list)):
            yield from _iter_jsonld_nodes(v)


def _types_of(node: dict) -> list[str]:
    t = node.get("@type") or node.get("type")
    if t is None:
        return []
    if isinstance(t, list):
        return [str(x).lower() for x in t if x is not None]
    return [str(t).lower()]


def _node_keys_lower(node: dict) -> set[str]:
    return {str(k).lower() for k in node.keys()}


def schema_factors(soup, html: str) -> dict[str, float]:
    distinct_types: set[str] = set()
    same_as: set[str] = set()

    has_jsonld = False
    uses_org_localbiz = False
    uses_product_offer = False
    uses_agg_rating = False
    uses_faq = False
    uses_breadcrumb = False

    # ---- JSON-LD -------------------------------------------------------------
    try:
        scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)})
        for sc in scripts:
            has_jsonld = True
            raw = sc.string
            if raw is None:
                raw = sc.get_text()
            if not raw or not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            for node in _iter_jsonld_nodes(data):
                if not isinstance(node, dict):
                    continue
                keys = _node_keys_lower(node)
                types = _types_of(node)
                for ty in types:
                    distinct_types.add(ty)
                tset = set(types)
                if tset & _ORG_LOCALBIZ:
                    uses_org_localbiz = True
                if tset & _PRODUCT_OFFER or (keys & _OFFER_PRICE_KEYS):
                    uses_product_offer = True
                if tset & _AGG_RATING or (keys & _RATING_KEYS):
                    uses_agg_rating = True
                if tset & _FAQ:
                    uses_faq = True
                if tset & _BREADCRUMB:
                    uses_breadcrumb = True
                # sameAs -> claimed brands
                sa = node.get("sameAs")
                if sa is not None:
                    if isinstance(sa, str):
                        same_as.add(sa.strip())
                    elif isinstance(sa, list):
                        for u in sa:
                            if isinstance(u, str) and u.strip():
                                same_as.add(u.strip())
    except Exception:
        pass

    # ---- Microdata -----------------------------------------------------------
    try:
        for el in soup.find_all(attrs={"itemtype": True}):
            itemtype = el.get("itemtype")
            if isinstance(itemtype, list):
                itemtype = " ".join(itemtype)
            if not itemtype:
                continue
            # itemtype is a URL like https://schema.org/Product — take last path seg
            for piece in re.split(r"\s+", itemtype.strip()):
                ty = piece.rstrip("/").rsplit("/", 1)[-1].lower()
                if not ty:
                    continue
                distinct_types.add(ty)
                if ty in _ORG_LOCALBIZ:
                    uses_org_localbiz = True
                if ty in _PRODUCT_OFFER:
                    uses_product_offer = True
                if ty in _AGG_RATING:
                    uses_agg_rating = True
                if ty in _FAQ:
                    uses_faq = True
                if ty in _BREADCRUMB:
                    uses_breadcrumb = True

        # microdata itemprop signals (price / ratingValue / sameAs)
        for el in soup.find_all(attrs={"itemprop": True}):
            prop = el.get("itemprop")
            if isinstance(prop, list):
                prop = " ".join(prop)
            if not prop:
                continue
            plow = prop.strip().lower()
            if plow in _OFFER_PRICE_KEYS:
                uses_product_offer = True
            if plow in _RATING_KEYS:
                uses_agg_rating = True
            if plow == "sameas":
                href = el.get("href") or el.get("content")
                if isinstance(href, str) and href.strip():
                    same_as.add(href.strip())
    except Exception:
        pass

    return {
        "HAS_JSONLD": 1.0 if has_jsonld else 0.0,
        "SCHEMA_TYPES": float(len(distinct_types)),
        "USES_ORG_LOCALBIZ": 1.0 if uses_org_localbiz else 0.0,
        "USES_PRODUCT_OFFER": 1.0 if uses_product_offer else 0.0,
        "USES_AGG_RATING": 1.0 if uses_agg_rating else 0.0,
        "USES_FAQ": 1.0 if uses_faq else 0.0,
        "USES_BREADCRUMB": 1.0 if uses_breadcrumb else 0.0,
        "CLAIMED_BRANDS": float(len(same_as)),
    }


# property keys that are plumbing, not meaningful entity attributes
_TRIPLE_SKIP_KEYS = {"@type", "@context", "@id", "@graph", "url", "image", "name"}
# per-node cap so a giant node can't flood the triple list
_MAX_TRIPLES_PER_NODE = 12


def schema_entities(soup, html: str) -> tuple[list[dict], list[dict]]:
    """Harvest author-declared entities + EAV triples from JSON-LD nodes.

    Returns (entities, triples):
      entities: [{"name": str, "type": str, "salience": 1.0}, ...]
      triples:  [{"entity": str, "attribute": str, "value": str, "is_edge": bool}, ...]
    """
    entities: list[dict] = []
    triples: list[dict] = []

    try:
        scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)})
        for sc in scripts:
            raw = sc.string
            if raw is None:
                raw = sc.get_text()
            if not raw or not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            for node in _iter_jsonld_nodes(data):
                if not isinstance(node, dict):
                    continue
                name = node.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                node_name = name.strip()

                # ---- author-declared entity --------------------------------
                types = _types_of(node)
                entities.append(
                    {
                        "name": node_name,
                        "type": types[0] if types else "",
                        "salience": 1.0,
                    }
                )

                # ---- scalar attribute triples + edges ----------------------
                emitted = 0
                for k, v in node.items():
                    key = str(k)
                    klow = key.lower()
                    if klow in _TRIPLE_SKIP_KEYS:
                        continue

                    # sameAs -> edge(s) to external authority URLs
                    if klow == "sameas":
                        sa_vals = v if isinstance(v, list) else [v]
                        for sv in sa_vals:
                            if isinstance(sv, str) and sv.strip():
                                triples.append(
                                    {
                                        "entity": node_name,
                                        "attribute": "sameas",
                                        "value": sv.strip(),
                                        "is_edge": True,
                                    }
                                )
                        continue

                    # nested node-with-a-name -> edge (e.g. brand, manufacturer)
                    if isinstance(v, dict):
                        target = v.get("name")
                        if isinstance(target, str) and target.strip():
                            triples.append(
                                {
                                    "entity": node_name,
                                    "attribute": klow,
                                    "value": target.strip(),
                                    "is_edge": True,
                                }
                            )
                        continue
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                target = item.get("name")
                                if isinstance(target, str) and target.strip():
                                    triples.append(
                                        {
                                            "entity": node_name,
                                            "attribute": klow,
                                            "value": target.strip(),
                                            "is_edge": True,
                                        }
                                    )
                        continue

                    # scalar attribute -> plain triple (cap per node)
                    if isinstance(v, bool):
                        continue  # bools are rarely meaningful EAV values
                    if isinstance(v, (str, int, float)):
                        sval = str(v).strip()
                        if not sval:
                            continue
                        if emitted >= _MAX_TRIPLES_PER_NODE:
                            continue
                        triples.append(
                            {
                                "entity": node_name,
                                "attribute": klow,
                                "value": sval,
                                "is_edge": False,
                            }
                        )
                        emitted += 1
    except Exception:
        return ([], [])

    return (entities, triples)
