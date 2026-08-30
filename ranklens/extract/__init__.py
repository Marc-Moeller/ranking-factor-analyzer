"""The extraction layer — HTML + SERP + corpus -> the ~65 ranking factors.

Public API:
- ``extract_html_factors(html, url, domain, vset, load_ms) -> (factors, body_text)``
- ``schema_factors(soup, html) -> dict``
- ``serp_factors(item, vset) -> dict``
- ``build_corpus(body_texts, vset) -> Corpus`` / ``corpus_factors(...)`` / ``Corpus``
"""
from __future__ import annotations

from .corpus import Corpus, build_corpus, corpus_factors
from .factors import extract_html_factors
from .schema import schema_factors
from .serp_factors import serp_factors

__all__ = [
    "extract_html_factors",
    "schema_factors",
    "serp_factors",
    "build_corpus",
    "corpus_factors",
    "Corpus",
]
