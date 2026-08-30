"""Keyword variation set — the unit Cora counts everywhere ("Variations in X").

A "variation" is any token / n-gram derived from the target keyword. Counting a
factor like "Variations in H1 Tags" means: total occurrences of any variation in
the H1 text. This module is shared by the page-factor extractor and the
SERP-presentation factor computation so both count identically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "at",
    "by", "from", "is", "it", "as", "be", "are", "your", "you", "best", "top",
}
_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class VariationSet:
    keyword: str
    phrase: str                              # normalized exact phrase
    variations: tuple[str, ...] = field(default_factory=tuple)  # all variation strings
    _patterns: tuple = field(default_factory=tuple, repr=False)

    def count(self, text: str) -> int:
        """Total occurrences of any variation in text (case-insensitive)."""
        if not text:
            return 0
        low = text.lower()
        return sum(len(p.findall(low)) for p in self._patterns)

    def count_exact(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(re.escape(self.phrase), text.lower()))

    def unique_used(self, text: str) -> int:
        """How many DISTINCT variations appear at least once."""
        if not text:
            return 0
        low = text.lower()
        return sum(1 for p in self._patterns if p.search(low))

    def starts_with_variation(self, text: str) -> bool:
        if not text:
            return False
        low = text.strip().lower()
        return any(low.startswith(v) for v in self.variations)

    def first_match_offset(self, text: str) -> int | None:
        """Byte offset of the first variation occurrence, or None."""
        if not text:
            return None
        low = text.lower()
        best: int | None = None
        for p in self._patterns:
            m = p.search(low)
            if m and (best is None or m.start() < best):
                best = m.start()
        return best


def build_variation_set(keyword: str) -> VariationSet:
    kw = keyword.strip().lower()
    toks = _tokens(kw)
    phrase = " ".join(toks)

    variations: set[str] = set()
    if phrase:
        variations.add(phrase)
    content_toks = [t for t in toks if t not in _STOP and len(t) > 2]
    variations.update(content_toks)
    # contiguous bigrams + trigrams of the full token list
    for n in (2, 3):
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i:i + n])
            variations.add(gram)
    # phrase without stopwords (e.g. "buy running shoes" already, but helps longer kws)
    if content_toks:
        variations.add(" ".join(content_toks))

    variations = {v for v in variations if v}
    # longest first so multi-word grams match before single tokens when iterating
    ordered = tuple(sorted(variations, key=lambda s: (-len(s), s)))
    patterns = tuple(re.compile(r"\b" + re.escape(v) + r"\b") for v in ordered)
    return VariationSet(keyword=keyword, phrase=phrase, variations=ordered, _patterns=patterns)


def word_count(text: str) -> int:
    return len(_tokens(text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]
