"""Passage-level semantic relevance and SERP-corpus salient-term coverage."""
from __future__ import annotations

from collections import Counter
import json
import math
import re

import numpy as np

from ranklens.clients.embeddings import embed_texts
from ranklens.clients.entities import _first_balanced_object, _strip_fences
from ranklens.clients.llm import chat, llm_unavailable
from ranklens.config import Settings, get_settings
from ranklens.models import SemanticReport, SerpItem, SubIntent

PASSAGE_MIN_WORDS = 120  # Enough context to represent a coherent answer passage.
PASSAGE_MAX_WORDS = 180  # Keeps long sections from diluting passage relevance.
MAX_PASSAGES_PER_PAGE = 60  # Bounds embedding cost and large-page influence.
# Similarity thresholds are method-dependent: embedding cosines for related
# text sit around 0.4-0.8, while sparse TF-IDF cosines between a short query
# and a 150-word passage rarely exceed 0.3 even on-topic. One shared threshold
# would zero out coverage whenever the TF-IDF fallback runs.
SUBINTENT_MATCH_THRESHOLD = {"embeddings": 0.45, "tfidf": 0.15}
OPEN_GAP_THRESHOLD = {"embeddings": 0.35, "tfidf": 0.08}
MAX_SALIENT_TERMS = 40  # A compact corpus vocabulary remains actionable.
CHAT_COST_USD = 0.0012

_WORD_RE = re.compile(r"[a-z][a-z0-9']*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "may", "me", "more", "most", "my", "no", "not", "of",
    "on", "one", "or", "our", "out", "over", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "up", "us", "use",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "why", "will", "with", "would", "you", "your",
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def _pack_sentences(text: str) -> list[str]:
    passages: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _sentences(text) or [text.strip()]:
        words = sentence.split()
        while len(words) > PASSAGE_MAX_WORDS:
            if current:
                passages.append(" ".join(current))
                current, count = [], 0
            passages.append(" ".join(words[:PASSAGE_MAX_WORDS]))
            words = words[PASSAGE_MAX_WORDS:]
        if current and count + len(words) > PASSAGE_MAX_WORDS:
            passages.append(" ".join(current))
            current, count = [], 0
        current.extend(words)
        count += len(words)
        if count >= PASSAGE_MIN_WORDS:
            passages.append(" ".join(current))
            current, count = [], 0
    if current:
        if passages and len(current) < PASSAGE_MIN_WORDS // 2:
            combined = f"{passages[-1]} {' '.join(current)}"
            words = combined.split()
            if len(words) <= PASSAGE_MAX_WORDS:
                passages[-1] = combined
            else:
                passages.append(" ".join(current))
        else:
            passages.append(" ".join(current))
    return passages


def _passages(body: str) -> list[str]:
    """Split visible body text into bounded, paragraph-aware passages."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body or "") if p.strip()]
    if len(paragraphs) < 2:
        return _pack_sentences(body)[:MAX_PASSAGES_PER_PAGE]

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph.split()) > PASSAGE_MAX_WORDS:
            units.extend(_pack_sentences(paragraph))
        else:
            units.append(paragraph)
    passages: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = unit.split()
        if current and count + len(words) > PASSAGE_MAX_WORDS:
            passages.append("\n\n".join(current))
            current, count = [], 0
        current.append(unit)
        count += len(words)
        if count >= PASSAGE_MIN_WORDS:
            passages.append("\n\n".join(current))
            current, count = [], 0
    if current:
        passages.append("\n\n".join(current))
    return passages[:MAX_PASSAGES_PER_PAGE]


def _tokens(text: str) -> list[str]:
    return [token for token in _WORD_RE.findall((text or "").lower()) if token not in _STOPWORDS]


def _features(text: str) -> list[str]:
    tokens = _tokens(text)
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _tfidf_matrix(texts: list[str]) -> np.ndarray:
    documents = [Counter(_features(text)) for text in texts]
    vocabulary = sorted({term for document in documents for term in document})
    if not vocabulary:
        return np.zeros((len(texts), 0), dtype=float)
    index = {term: i for i, term in enumerate(vocabulary)}
    document_frequency = Counter(term for document in documents for term in document)
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=float)
    for row, document in enumerate(documents):
        for term, frequency in document.items():
            tf = 1.0 + math.log(frequency)
            idf = math.log((1.0 + len(documents)) / (1.0 + document_frequency[term])) + 1.0
            matrix[row, index[term]] = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _normalized_matrix(vectors: list[list[float]]) -> np.ndarray | None:
    try:
        matrix = np.asarray(vectors, dtype=float)
        if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
            return None
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    except Exception:  # noqa: BLE001 — malformed provider output triggers fallback
        return None


def _first_balanced_array(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_sub_intents(reply: str, keyword: str) -> list[SubIntent]:
    cleaned = _strip_fences(reply)
    candidates = [cleaned, _first_balanced_array(cleaned), _first_balanced_object(cleaned)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("sub_intents") or parsed.get("intents") or []
        if not isinstance(parsed, list):
            continue
        intents: list[SubIntent] = []
        seen: set[str] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            key = name.casefold()
            if not name or key in seen:
                continue
            intents.append(SubIntent(name=name, description=str(item.get("description") or "").strip()))
            seen.add(key)
            if len(intents) == 10:
                break
        if intents:
            return intents
    return [SubIntent(name=keyword, description="The searcher's primary query.")]


async def _fan_out(keyword: str, settings: Settings) -> list[SubIntent]:
    messages = [
        {
            "role": "system",
            "content": (
                "Decompose a search query into 5-10 distinct sub-intents a useful page should cover. "
                "Return only a JSON list of objects with string fields name and description."
            ),
        },
        {"role": "user", "content": f"Search query: {keyword}"},
    ]
    # Two attempts: the shared provider stack 429s under the concurrent funnel
    # load, and a failed fan-out degrades the whole coverage matrix to one row.
    for _ in range(2):
        try:
            reply = await chat(messages, max_tokens=700, temperature=0.0, settings=settings)
            if llm_unavailable(reply):
                continue
            intents = _parse_sub_intents(reply, keyword)
            if len(intents) > 1 or intents[0].name != keyword:
                return intents
        except Exception:  # noqa: BLE001 — bare keyword keeps semantic analysis usable
            continue
    return [SubIntent(name=keyword, description="The searcher's primary query.")]


def _salient_terms(keyword: str, bodies_by_rank: dict[int, str]) -> list[str]:
    ranked = [bodies_by_rank[rank] for rank in sorted(bodies_by_rank) if 1 <= rank <= 10]
    keywordless_documents = [Counter(_features(body)) for body in ranked if body.strip()]
    if not keywordless_documents:
        return []
    keyword_tokens = set(_tokens(keyword))
    document_frequency = Counter(term for doc in keywordless_documents for term in doc)
    scores: dict[str, float] = {}
    for term in document_frequency:
        parts = term.split()
        if any(part.isdigit() for part in parts) or keyword_tokens.intersection(parts):
            continue
        weighted_tf = sum(1.0 + math.log(doc[term]) for doc in keywordless_documents if doc[term])
        scores[term] = weighted_tf * math.log1p(document_frequency[term])
    return [term for term, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:MAX_SALIENT_TERMS]]


async def analyze_semantic(
    keyword: str,
    serp_items: list[SerpItem],
    bodies_by_rank: dict[int, str],
    settings: Settings | None = None,
) -> tuple[SemanticReport | None, dict[int, dict[str, float]]]:
    """Analyze page passages against the query and its searcher sub-intents.

    Args:
        keyword: Search query being analyzed.
        serp_items: Ranked SERP results; accepted for the shared pipeline contract.
        bodies_by_rank: Visible body text keyed by rank, with target at rank 0.
        settings: Optional pre-loaded runtime settings.

    Returns:
        The semantic panel and per-rank factor values, or ``(None, {})`` on an
        unrecoverable failure. Never raises.
    """
    try:
        del serp_items
        settings = settings or get_settings()
        keyword = (keyword or "").strip()
        if not keyword:
            return None, {}
        sub_intents = await _fan_out(keyword, settings)
        passages_by_rank = {
            int(rank): _passages(body)
            for rank, body in bodies_by_rank.items()
            if body and body.strip()
        }
        passages_by_rank = {rank: passages for rank, passages in passages_by_rank.items() if passages}

        query_texts = [keyword] + [
            f"{intent.name}. {intent.description}".strip() for intent in sub_intents
        ]
        passage_rows = [(rank, passage) for rank in sorted(passages_by_rank) for passage in passages_by_rank[rank]]
        all_texts = query_texts + [passage for _, passage in passage_rows]
        vectors = await embed_texts(all_texts, settings)
        matrix = _normalized_matrix(vectors) if vectors is not None else None
        method = "embeddings"
        if matrix is None or matrix.shape[0] != len(all_texts):
            matrix = _tfidf_matrix(all_texts)
            method = "tfidf"

        query_vectors = matrix[: len(query_texts)]
        passage_vectors = matrix[len(query_texts) :]
        coverage: dict[int, dict[str, float]] = {}
        best_passage_sim: dict[int, float] = {}
        content_focus: dict[int, float] = {}
        target_best_passage = ""
        offset = 0
        for rank in sorted(passages_by_rank):
            page_passages = passages_by_rank[rank]
            page_vectors = passage_vectors[offset : offset + len(page_passages)]
            offset += len(page_passages)
            raw_similarities = np.clip(page_vectors @ query_vectors[0], 0.0, 1.0)
            best_index = int(np.argmax(raw_similarities))
            best_passage_sim[rank] = float(raw_similarities[best_index])
            content_focus[rank] = float(np.mean(raw_similarities))
            coverage[rank] = {}
            for intent_index, intent in enumerate(sub_intents, start=1):
                similarities = np.clip(page_vectors @ query_vectors[intent_index], 0.0, 1.0)
                coverage[rank][intent.name] = float(np.max(similarities))
            if rank == 0:
                target_best_passage = page_passages[best_index][:300]

        match_threshold = SUBINTENT_MATCH_THRESHOLD[method]
        gap_threshold = OPEN_GAP_THRESHOLD[method]
        # With only the bare-keyword fallback intent there is no fan-out to
        # find gaps in — an "open gap" on the query itself is noise.
        open_gaps = [
            intent.name for intent in sub_intents
            if max((page.get(intent.name, 0.0) for page in coverage.values()), default=0.0)
            < gap_threshold
        ] if len(sub_intents) > 1 else []
        salient_terms = _salient_terms(keyword, bodies_by_rank)
        factors: dict[int, dict[str, float]] = {}
        for rank in sorted(passages_by_rank):
            body_features = set(_features(bodies_by_rank.get(rank, "")))
            salient_coverage = (
                100.0 * sum(term in body_features for term in salient_terms) / len(salient_terms)
                if salient_terms else 0.0
            )
            covered = sum(
                similarity >= match_threshold for similarity in coverage[rank].values()
            )
            factors[rank] = {
                "BEST_PASSAGE_SIM": round(best_passage_sim[rank] * 100.0, 4),
                "SUBINTENT_COVERAGE": round(100.0 * covered / len(sub_intents), 4),
                "CONTENT_FOCUS": round(content_focus[rank] * 100.0, 4),
                "SALIENT_TERM_COVERAGE": round(salient_coverage, 4),
            }

        summary = (
            f"Analyzed {len(passages_by_rank)} pages across {len(sub_intents)} "
            f"search intents using {method.upper()} passage matching."
        )
        report = SemanticReport(
            sub_intents=sub_intents,
            method=method,
            coverage=coverage,
            best_passage_sim=best_passage_sim,
            content_focus=content_focus,
            target_best_passage=target_best_passage,
            open_gaps=open_gaps,
            summary=summary,
            cost_usd=CHAT_COST_USD,
        )
        return report, factors
    except Exception:  # noqa: BLE001 — semantic enrichment must never fail a run
        return None, {}
