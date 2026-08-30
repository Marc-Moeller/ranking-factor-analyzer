"""Corpus-derived factors — LSI terms + TF/IDF over the fetched page set.

Cora derives LSI words and IDF from the 100-page ranking corpus. We approximate:

* Build a document-frequency table over the cleaned body texts.
* LSI terms = mid-frequency terms (appear in 25%–90% of docs), excluding
  stopwords and any token belonging to the keyword variation set; ranked by
  document frequency, top 150 kept.
* IDF(term) = log(N / df).

``corpus_factors`` then emits TF, TFIDF, LSI_SENTENCES, UNIQUE_LSI for one page.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# reuse the exact tokenizer the keyword module uses
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# small, self-contained stoplist (kept here so corpus has no external dep)
_STOPLIST = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "at",
    "by", "from", "is", "it", "as", "be", "are", "your", "you", "best", "top",
    "this", "that", "these", "those", "i", "we", "they", "he", "she", "him",
    "her", "his", "our", "us", "my", "me", "their", "them", "its", "was", "were",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "must", "shall", "not", "no",
    "but", "if", "then", "than", "so", "too", "very", "just", "out", "up",
    "down", "off", "over", "under", "again", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "what", "which", "who", "whom", "when", "where", "why", "how", "about",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "there", "here", "also", "get", "got", "one", "two", "new", "now", "us",
    "use", "using", "used", "like", "make", "made", "need", "want", "see",
    "way", "well", "back", "even", "much", "many", "per", "via", "etc",
}


class Corpus:
    def __init__(self, n_docs: int, df: Counter, lsi_terms: list[str]):
        self.n_docs = max(int(n_docs), 0)
        self._df = df
        self.lsi_terms: list[str] = lsi_terms
        self._lsi_set = set(lsi_terms)

    def idf(self, term: str) -> float:
        if not term:
            return 0.0
        df = self._df.get(term.lower(), 0)
        if df <= 0 or self.n_docs <= 0:
            # unseen term -> max-ish idf based on smoothing
            return math.log((self.n_docs + 1) / 1.0) if self.n_docs > 0 else 0.0
        return math.log(self.n_docs / df)

    def count_lsi(self, text: str) -> int:
        """Total occurrences of any LSI term in text."""
        if not self._lsi_set or not text:
            return 0
        toks = _tokens(text)
        return sum(1 for t in toks if t in self._lsi_set)

    def unique_lsi(self, text: str) -> int:
        """Distinct LSI terms present in text."""
        if not self._lsi_set or not text:
            return 0
        toks = set(_tokens(text))
        return len(toks & self._lsi_set)


def build_corpus(body_texts: list[str], vset) -> Corpus:
    docs_tokens: list[set[str]] = []
    df: Counter = Counter()
    for t in body_texts or []:
        toks = set(_tokens(t))
        docs_tokens.append(toks)
        for tok in toks:
            df[tok] += 1

    n = len(docs_tokens)
    if n == 0:
        return Corpus(0, Counter(), [])

    lo = 0.25 * n
    hi = 0.90 * n

    # tokens that are part of the keyword variation set (any whitespace-split word)
    var_tokens: set[str] = set()
    try:
        for v in getattr(vset, "variations", ()):
            for w in str(v).split():
                var_tokens.add(w.lower())
    except Exception:
        pass

    candidates: list[tuple[str, int]] = []
    for term, freq in df.items():
        if freq < lo or freq > hi:
            continue
        if term in _STOPLIST:
            continue
        if term in var_tokens:
            continue
        if len(term) < 3:
            continue
        candidates.append((term, freq))

    # rank by df desc, then alphabetical for stability; keep top 150
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    lsi_terms = [term for term, _ in candidates[:150]]

    return Corpus(n, df, lsi_terms)


def corpus_factors(clean_body_text: str, vset, corpus: Corpus) -> dict[str, float]:
    out: dict[str, float] = {"TF": 0.0, "TFIDF": 0.0, "LSI_SENTENCES": 0.0, "UNIQUE_LSI": 0.0}

    try:
        wc = max(len(_tokens(clean_body_text)), 1)
        raw_matches = vset.count(clean_body_text)
        tf = raw_matches / wc * 1000.0
        out["TF"] = float(tf)
    except Exception:
        out["TF"] = 0.0
        raw_matches = 0
        wc = 1

    try:
        kw_tokens = [w for w in (vset.phrase or "").split() if w]
        if kw_tokens and corpus is not None:
            mean_idf = sum(corpus.idf(t) for t in kw_tokens) / len(kw_tokens)
        else:
            mean_idf = 0.0
        out["TFIDF"] = float(out["TF"] * mean_idf)
    except Exception:
        out["TFIDF"] = 0.0

    try:
        out["LSI_SENTENCES"] = float(corpus.count_lsi(clean_body_text)) if corpus else 0.0
    except Exception:
        out["LSI_SENTENCES"] = 0.0

    try:
        out["UNIQUE_LSI"] = float(corpus.unique_lsi(clean_body_text)) if corpus else 0.0
    except Exception:
        out["UNIQUE_LSI"] = 0.0

    return out
