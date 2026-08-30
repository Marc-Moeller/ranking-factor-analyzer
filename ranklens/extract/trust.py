"""Deterministic trust and transparency signals extracted from raw HTML."""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup


_TRUST_IDS = (
    "TRUST_BYLINE",
    "TRUST_AUTHOR_PAGE",
    "TRUST_PERSON_SCHEMA",
    "TRUST_ORG_SCHEMA",
    "TRUST_SAMEAS",
    "TRUST_OUTBOUND_CITATIONS",
    "TRUST_CONTACT_LINK",
    "TRUST_VISIBLE_DATE",
)
_SOCIAL_DOMAINS = (
    "twitter.com", "x.com", "facebook.com", "linkedin.com",
    "instagram.com", "youtube.com", "pinterest.com",
)
_CONTACT_RE = re.compile(r"(?:contact|about|impressum)", re.I)
_BYLINE_RE = re.compile(r"\bby\s+[A-Z][\w.'’-]+(?:\s+[A-Z][\w.'’-]+){0,3}\b")


def _registrable_domain(host: str) -> str:
    host = (host or "").lower().split(":", 1)[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return ".".join(parts)
    if len(parts[-2]) <= 3 and len(parts[-1]) <= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _jsonld_nodes(value):
    if isinstance(value, list):
        for item in value:
            yield from _jsonld_nodes(item)
    elif isinstance(value, dict):
        yield value
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from _jsonld_nodes(item)


def _types(node: dict) -> set[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {str(value).lower() for value in values if value}


def trust_factors(html: str, url: str) -> dict[str, float]:
    """Extract all registered trust factors from a page.

    Args:
        html: Raw page HTML, which may contain malformed markup or JSON-LD.
        url: Absolute page URL used to distinguish internal and external links.

    Returns:
        A complete ``TRUST_*`` factor mapping. Missing signals are zero and the
        function never raises.
    """
    factors = {factor_id: 0.0 for factor_id in _TRUST_IDS}
    try:
        try:
            soup = BeautifulSoup(html or "", "lxml")
        except Exception:
            soup = BeautifulSoup(html or "", "html.parser")

        page_domain = _registrable_domain(urlsplit(url or "").hostname or "")
        nodes: list[dict] = []
        for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
            try:
                parsed = json.loads(script.string or script.get_text() or "")
                nodes.extend(node for node in _jsonld_nodes(parsed) if isinstance(node, dict))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

        factors["TRUST_PERSON_SCHEMA"] = 1.0 if any("person" in _types(node) for node in nodes) else 0.0
        factors["TRUST_ORG_SCHEMA"] = 1.0 if any(
            _types(node) & {"organization", "corporation", "localbusiness"} for node in nodes
        ) else 0.0

        same_as: set[str] = set()
        for node in nodes:
            values = node.get("sameAs", [])
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and urlsplit(value).hostname:
                        if _registrable_domain(urlsplit(value).hostname or "") != page_domain:
                            same_as.add(value)
        factors["TRUST_SAMEAS"] = float(len(same_as))

        visible_text = soup.get_text(" ", strip=True)
        author_meta = soup.find("meta", attrs={
            "name": re.compile(r"^(?:author|article:author)$", re.I)
        }) or soup.find("meta", attrs={"property": re.compile(r"^article:author$", re.I)})
        rel_author = soup.find("a", rel=lambda value: value and "author" in value)
        author_class = soup.find(class_=re.compile(r"(?:^|[-_])author(?:[-_]|$)", re.I))
        has_byline = bool(rel_author or author_class or author_meta or _BYLINE_RE.search(visible_text))
        factors["TRUST_BYLINE"] = 1.0 if has_byline else 0.0

        author_link = rel_author
        if not author_link and author_class:
            author_link = author_class if author_class.name == "a" else author_class.find("a", href=True)
        if not author_link:
            author_link = soup.find("a", href=re.compile(r"(?:author|about)", re.I))
        factors["TRUST_AUTHOR_PAGE"] = 1.0 if has_byline and author_link else 0.0

        citations = 0
        has_contact = False
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href") or "").strip()
            absolute = urljoin(url or "", href)
            host = (urlsplit(absolute).hostname or "").lower()
            domain = _registrable_domain(host)
            label = f"{href} {anchor.get_text(' ', strip=True)}"
            if domain == page_domain and _CONTACT_RE.search(label):
                has_contact = True
            if not host or domain == page_domain:
                continue
            if any(host == social or host.endswith(f".{social}") for social in _SOCIAL_DOMAINS):
                continue
            citations += 1
        factors["TRUST_OUTBOUND_CITATIONS"] = float(min(citations, 50))
        factors["TRUST_CONTACT_LINK"] = 1.0 if has_contact else 0.0

        date_meta = soup.find("meta", attrs={
            "property": re.compile(r"^article:(?:published|modified)_time$", re.I)
        }) or soup.find("meta", attrs={
            "name": re.compile(r"^article:(?:published|modified)_time$", re.I)
        })
        schema_date = any(node.get("datePublished") or node.get("dateModified") for node in nodes)
        factors["TRUST_VISIBLE_DATE"] = 1.0 if soup.find("time", datetime=True) or date_meta or schema_date else 0.0
    except Exception:
        pass
    return factors
