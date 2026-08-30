"""Entity / EAV knowledge-graph extraction from one page's text, via the configured OpenAI-compatible LLM endpoint.

A single cheap LLM call turns a page's body text into three arrays — the
entities the page is about, the attribute->value claims it asserts, and the
entity->entity relationships it implies — so the discovery layer can mine
entity coverage and EAV factors across a SERP.

Everything degrades gracefully, mirroring :mod:`ranklens.analyze.brand`: empty
input short-circuits without an LLM call, the model's reply is parsed
defensively (code fences / stray prose are tolerated), and any failure — the
``chat()`` ``"[AI report unavailable: ...]"`` sentinel, a parse error, or an
unexpected shape — yields ``{}`` rather than raising. The report renders without
the entity layer instead of failing the whole run.
"""
from __future__ import annotations

import asyncio
import json
import re

from ranklens.clients.llm import chat, llm_unavailable
from ranklens.config import Settings, get_settings

# Cost caps for the single extraction call.
MAX_INPUT_CHARS = 12000
MAX_TOKENS = 3000
MAX_ENTITIES = 40
# A whole page used to silently zero out — most visibly the tracked target, which
# then read "you cover 0 claims" even with a clear price table on the page. The
# extraction retries the OpenAI-compatible endpoint with rising temperature and
# stops the instant any entities/triples/edges come back. The temperature climb
# shakes loose the deterministic empty/truncation whiffs cheaper models are
# prone to. Plus: partial-JSON salvage (keep complete items from a truncated
# reply) and a roomy token cap so a dense price table doesn't overflow.
_FALLBACK_TEMPS = (0.0, 0.35, 0.7)

_SYSTEM_PROMPT = (
    "You are a knowledge-graph extractor. From one web page about \"{keyword}\", "
    "extract the REAL-WORLD facts a human would learn — never the document's CMS "
    "or SEO plumbing.\n\n"
    "Output ONE record per line. NO JSON, NO markdown, NO prose, no header, no "
    "blank lines. Each line is semicolon-separated fields, and the FIRST field is "
    "a one-letter tag:\n"
    "  E;<name>;<type>;<salience 0-1>          (an entity)\n"
    "  T;<entity>;<attribute>;<value>          (an attribute->value fact)\n"
    "  R;<entity>;<relation>;<target entity>   (an entity->entity relationship)\n"
    "Never use a semicolon inside a field. Keep attribute / relation names short, "
    "lowercase, spaces fine. Only assert what the page actually states.\n\n"
    "E (entities): specific, named real-world things the page is actually about — "
    "concrete products WITH their model name, brands, companies, people, places, "
    "materials, standards. At most 40, most central first. salience 0-1 = how "
    "central it is. GOOD: E;Herman Miller Aeron;product;0.9 . ALWAYS include the "
    "page's CORE SUBJECT — the main product, service, procedure, or topic the page "
    "is about (this is usually the thing named in the query \"{keyword}\", e.g. the "
    "procedure or product class itself) — as an entity with HIGH salience, even "
    "when it is a general category. BAD — never emit page sections, headings, "
    "navigation labels, Home, Article, Blog, Buying Guide, FAQ, the article's own "
    "headline, the website name, or generic categories that are NOT the page's "
    "core subject.\n\n"
    "T (facts): a concrete attribute->value claim about a SPECIFIC named entity; "
    "value is the literal asserted fact, short. GOOD: T;Herman Miller Aeron;"
    "release date;1 Sep 2025  /  T;single dental implant;starting price;from "
    "$3,999 . Cover price, weight capacity, warranty, material, dimensions, max "
    "load, certifications, country of origin. BAD — NEVER emit document/SEO "
    "metadata: datePublished, dateModified, inLanguage, url, headline, wordCount, "
    "articleSection, breadcrumb position, list item index, the article's author, "
    "the website publisher, page slug. If a fact is about the document rather "
    "than the real-world thing, drop it.\n\n"
    "R (relationships): target is ANOTHER named entity. GOOD: R;Elon Musk;married "
    "to;Talulah Riley . Also manufactured by, owned by, founded by, subsidiary "
    "of, located in, successor of, compatible with. BAD: sameAs links to social "
    "profiles, next/previous list pointers, or any target that is a URL or the "
    "document itself."
)


def _clamp01(value) -> float:
    """Coerce ``value`` to a float clamped to ``0.0..1.0`` (default ``0.0``)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if out != out:  # NaN
        return 0.0
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _strip_fences(text: str) -> str:
    """Drop a leading/trailing ```` ```json ```` code fence if present."""
    s = (text or "").strip()
    if s.startswith("```"):
        # Remove the opening fence line (```` ``` ```` or ```` ```json ````).
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", s, count=1)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _first_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring of ``text``, or ``None``.

    Brace-counts while skipping over braces inside double-quoted strings so a
    ``{`` or ``}`` in a value doesn't throw off the depth.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _salvage_arrays(text: str) -> dict:
    """Recover ``entities`` / ``triples`` / ``edges`` from a truncated reply.

    When the outer object never closes (the model hit the token cap mid-array),
    a strict parse yields nothing. Here we locate each ``"key": [`` and pull out
    every fully-balanced ``{...}`` object that follows, stopping at the array's
    ``]`` or end-of-text. A half-written trailing object is simply dropped, so we
    keep all the complete claims the model did manage to emit instead of losing
    the whole page.
    """
    out: dict = {}
    for key in ("entities", "triples", "edges"):
        m = re.search(r'"' + key + r'"\s*:\s*\[', text)
        if not m:
            continue
        items: list = []
        i = m.end()
        while i < len(text):
            while i < len(text) and text[i] in " \t\r\n,":
                i += 1
            if i >= len(text) or text[i] == "]":
                break
            if text[i] != "{":
                break
            obj = _first_balanced_object(text[i:])
            if not obj:
                break  # truncated final object — drop it, keep the rest
            try:
                items.append(json.loads(obj))
            except (json.JSONDecodeError, ValueError):
                pass
            i += len(obj)
        if items:
            out[key] = items
    return out


def _parse_json_object(text: str) -> dict:
    """Parse the model reply into a dict, tolerating fences / stray prose.

    Tries a direct ``json.loads`` of the de-fenced text first, then the first
    balanced ``{...}`` substring, then — for a reply truncated mid-array — a
    best-effort array salvage. Returns ``{}`` if nothing usable parses.
    """
    cleaned = _strip_fences(text)
    for candidate in (cleaned, _first_balanced_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return _salvage_arrays(cleaned)


def _parse_toon(text: str) -> dict:
    """Parse the compact tag;field;field line format into the frozen shape.

    One record per line: ``E;name;type;salience`` / ``T;entity;attribute;value``
    / ``R;entity;relation;target``. Robust by construction — a truncated final
    line just gets dropped, and there are no nested delimiters to unbalance. As a
    courtesy we also accept an untagged ``entity;attribute;value`` line as a fact.
    """
    entities: list[dict] = []
    triples: list[dict] = []
    edges: list[dict] = []
    for raw in _strip_fences(text).replace("|", "\n").splitlines():
        line = raw.strip().lstrip("-*• \t")
        if not line or ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        tag = parts[0].upper()
        if tag == "E" and len(parts) >= 4:
            entities.append({"name": parts[1], "type": parts[2], "salience": parts[3]})
        elif tag == "T" and len(parts) >= 4:
            # value = parts[3:] joined so a stray ';' inside a value survives.
            triples.append({"entity": parts[1], "attribute": parts[2], "value": ";".join(parts[3:])})
        elif tag == "R" and len(parts) >= 4:
            edges.append({"entity": parts[1], "relation": parts[2], "target": ";".join(parts[3:])})
        elif tag not in {"E", "T", "R"} and len(parts) >= 3:
            # Untagged fallback: treat as entity;attribute;value.
            triples.append({"entity": parts[0], "attribute": parts[1], "value": ";".join(parts[2:])})
    return {"entities": entities, "triples": triples, "edges": edges}


def _parse_extraction(text: str) -> dict:
    """Parse a reply into ``{entities, triples, edges}`` — TOON first, JSON next.

    The prompt asks for the compact line format, so try that first; if it yields
    nothing usable (a model that ignored the format and emitted JSON anyway), fall
    back to the tolerant JSON parser. Either way ``_normalize`` finishes the job.
    """
    toon = _parse_toon(text)
    if toon["entities"] or toon["triples"] or toon["edges"]:
        return toon
    return _parse_json_object(text)


def _normalize(parsed: dict) -> dict:
    """Coerce a parsed object to the frozen ``{entities, triples, edges}`` shape.

    Each list is rebuilt item-by-item with safe defaults; items missing a usable
    identity (entity ``name``; triple ``entity``+``value``; edge ``entity``+
    ``target``) are dropped. ``entities`` is capped at :data:`MAX_ENTITIES`.
    """
    entities: list[dict] = []
    for item in parsed.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        entities.append(
            {
                "name": name,
                "type": str(item.get("type") or ""),
                "salience": _clamp01(item.get("salience")),
            }
        )
        if len(entities) >= MAX_ENTITIES:
            break

    triples: list[dict] = []
    for item in parsed.get("triples") or []:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        value = str(item.get("value") or "").strip()
        if not entity or not value:
            continue
        triples.append(
            {
                "entity": entity,
                "attribute": str(item.get("attribute") or ""),
                "value": value,
            }
        )

    edges: list[dict] = []
    for item in parsed.get("edges") or []:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        target = str(item.get("target") or "").strip()
        if not entity or not target:
            continue
        edges.append(
            {
                "entity": entity,
                "relation": str(item.get("relation") or ""),
                "target": target,
            }
        )

    return {"entities": entities, "triples": triples, "edges": edges}


# Cap on how many distinct attributes we hand the canonicalizer in one call.
MAX_CANON_ATTRS = 160
MAX_CANON_TOKENS = 2000

_CANON_SYSTEM_PROMPT = (
    "You normalize attribute names that were extracted from SEVERAL web pages "
    "about \"{keyword}\" into ONE small shared vocabulary.\n\n"
    "You are given a list of raw attribute names (snake_case). Many are "
    "duplicates or near-synonyms for the SAME underlying property and must "
    "collapse onto a single canonical name. Examples of synonym groups:\n"
    "  cost_per_tooth, average_cost, dental_implant_price, consultation_cost, "
    "cost_per_arch, panoramic_x_ray_cost  -> price\n"
    "  boca_raton_address, miami_address, coral_springs_address  -> address\n"
    "  years_in_business, established, experience  -> experience\n"
    "  office_hours, hours  -> hours\n"
    "  average_lifespan, lifespan  -> lifespan\n"
    "  crown_material, material  -> material\n\n"
    "Rules:\n"
    "- Map EVERY input attribute to a canonical name. Output the input verbatim "
    "on the left.\n"
    "- The canonical name is short (1-2 words), generic, snake_case, with NO "
    "place names, brand names, or qualifiers (price, not average_implant_cost).\n"
    "- Prefer the simplest common term already present in the list.\n"
    "- Keep genuinely DISTINCT properties separate (price != warranty != phone).\n"
    "- An attribute that is already canonical maps to itself.\n\n"
    "Output ONE mapping per line, nothing else — no prose, no header, no blank "
    "lines:\n"
    "  raw_attribute => canonical_attribute"
)

_ARROW_RE = re.compile(r"\s*(?:=>|->|\t|:)\s*")


def _parse_attr_map(text: str, known: set[str], norm) -> dict[str, str]:
    """Parse ``raw => canonical`` lines into a normalized remap dict.

    Only entries whose left side is a KNOWN extracted attribute and whose
    canonical differs from it are kept (identity maps are dropped — they are
    no-ops). Both sides are run through ``norm`` (the same ``_norm_attr`` used at
    extraction time) so the keys line up with the triples we will rewrite.
    """
    mapping: dict[str, str] = {}
    for raw_line in _strip_fences(text).splitlines():
        line = raw_line.strip().lstrip("-*• \t")
        if not line:
            continue
        parts = _ARROW_RE.split(line, maxsplit=1)
        if len(parts) != 2:
            continue
        left = norm(parts[0])
        right = norm(parts[1])
        if not left or not right or left not in known or right == left:
            continue
        mapping[left] = right
    return mapping


async def canonicalize_attributes(
    attributes: list[str],
    keyword: str,
    *,
    settings: "Settings | None" = None,
) -> dict[str, str]:
    """Collapse near-synonym attribute names onto one shared canonical vocabulary.

    Takes the union of normalized attribute names extracted across all pages and
    returns ``{raw_attribute: canonical_attribute}`` for every name that should be
    renamed (identity maps omitted). This is the corpus-level pass that stops the
    EAV table fragmenting into ``cost_per_tooth`` / ``average_cost`` /
    ``dental_implant_price`` rows that all mean *price*.

    One cheap LLM call. Degrades to ``{}`` — an empty map means "rename
    nothing", so the table behaves exactly as before. Never raises.
    """
    # Dedup, keep first-seen order; need at least two distinct attrs to bother.
    attrs = list(dict.fromkeys(a.strip() for a in (attributes or []) if a and a.strip()))
    if len(attrs) < 2:
        return {}
    attrs = attrs[:MAX_CANON_ATTRS]

    try:
        settings = settings or get_settings()
        from ranklens.extract.entities import _norm_attr  # local: avoid import cycle

        messages = [
            {"role": "system", "content": _CANON_SYSTEM_PROMPT.format(keyword=keyword)},
            {"role": "user", "content": "Attributes:\n" + "\n".join(attrs)},
        ]

        try:
            reply = await chat(
                messages, model=settings.entity_model, max_tokens=MAX_CANON_TOKENS,
                temperature=0.0, settings=settings,
            )
        except Exception:  # noqa: BLE001 — treat as a flap
            reply = None

        if llm_unavailable(reply):
            return {}
        mapping = _parse_attr_map(reply, set(attrs), _norm_attr)
        return mapping if mapping else {}
    except Exception:  # noqa: BLE001 — best-effort normalization, never fatal
        return {}


async def extract_entities_llm(
    body_text: str,
    keyword: str,
    *,
    model: str | None = None,
    settings: "Settings | None" = None,
) -> dict:
    """Extract a page's entities + attribute/value triples + entity edges.

    One cheap, deterministic LLM call (temperature 0) through the configured
    OpenAI-compatible LLM endpoint. Input is truncated to :data:`MAX_INPUT_CHARS`
    to bound cost.

    Args:
        body_text: the page's visible body text. Empty/whitespace -> ``{}``.
        keyword: the query the SERP is about, for prompt context.
        model: override model id; defaults to ``settings.entity_model``.
        settings: optional pre-loaded settings.

    Returns:
        ``{"entities": [...], "triples": [...], "edges": [...]}`` normalized to
        the frozen shape, or ``{}`` on empty input or any failure. Never raises.
    """
    if not body_text or not body_text.strip():
        return {}

    try:
        settings = settings or get_settings()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(keyword=keyword)},
            {
                "role": "user",
                "content": (
                    f"Keyword: {keyword}\n\n"
                    "Page text:\n"
                    f"{body_text[:MAX_INPUT_CHARS]}"
                ),
            },
        ]

        # Rising temperature: a clean but empty reply escalates to a hotter
        # sample so cheap models that truncate or emit nothing get another try.
        for i, temp in enumerate(_FALLBACK_TEMPS):
            try:
                reply = await chat(
                    messages, model=model or settings.entity_model, max_tokens=MAX_TOKENS,
                    temperature=temp, settings=settings,
                )
            except Exception:  # noqa: BLE001 — treat as a flap, escalate
                reply = None

            ok = reply and not llm_unavailable(reply)
            if ok:
                result = _normalize(_parse_extraction(reply))
                if result["entities"] or result["triples"] or result["edges"]:
                    return result

            if i < len(_FALLBACK_TEMPS) - 1:
                await asyncio.sleep(0.5)

        return {}
    except Exception:  # noqa: BLE001 — entity layer is best-effort, never fatal
        return {}
