from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
from rabbithole import sources, vectors

SEARCH = "search"
CANDIDATES = "candidates"
RANKED = "ranked"
STAGES = (SEARCH, CANDIDATES, RANKED)
BAND_QUANTILE = 0.62
WEIGHTS = {"band": 0.30, "novelty": 0.15, "grounding": 0.35, "crowding": 0.20}

_JUNK_MARKERS = (
    "(disambiguation)", "list of ", "lists of ", "index of ", "outline of ",
    "glossary of ", "timeline of ", "bibliography of ", "comparison of ",)

_TITLE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "at", "by", "with",
    "from", "is", "are"})

_CATEGORY_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "century", "establishment",
    "establishments", "disestablishments", "introduction", "introductions", "article",
    "articles", "page", "pages", "wikipedia", "category", "stub", "stubs", "list",
    "lists", "people", "person", "born", "death", "deaths", "birth", "births", "year",
    "years"})


@dataclass
class Candidate:
    title: str
    sources: set[str] = field(default_factory=set)
    rank: float = 0.0
    score: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)


def title_tokens(title: str) -> set[str]:
    cleaned = title.lower().replace("(", " ").replace(")", " ").replace("-", " ")
    tokens = {token.strip(",.'\"") for token in cleaned.split()}
    return {
        token.rstrip("s") for token in tokens - _TITLE_STOPWORDS if len(token.rstrip("s")) > 2}


def is_junk(title: str) -> bool:
    lowered = title.lower()
    return any(marker in lowered for marker in _JUNK_MARKERS)


def is_lexical_variant(title: str, parent: str) -> bool:
    mine, theirs = title_tokens(title), title_tokens(parent)
    if not mine or not theirs:
        return False

    return mine <= theirs or theirs <= mine


def normalise_title(title: str) -> str:
    return " ".join(title.replace("_", " ").split()).casefold()


def candidates(parent: str, pool: int = 30, exclude: set[str] | None = None) -> list[Candidate]:
    blocked = {normalise_title(title) for title in (exclude or set())}
    blocked.add(normalise_title(parent))
    merged: dict[str, Candidate] = {}

    generators = {
        "morelike": [hit["title"] for hit in sources.morelike(parent, limit=pool)],
        "lead": sources.lead_links(parent)[:pool],}

    for source, titles in generators.items():
        for position, title in enumerate(titles):
            if is_junk(title) or is_lexical_variant(title, parent):
                continue
            if normalise_title(title) in blocked:
                continue

            candidate = merged.setdefault(title, Candidate(title=title))
            candidate.sources.add(source)
            candidate.rank += 1.0 / (60 + position)   # reciprocal rank fusion k=60

    return sorted(merged.values(), key=lambda c: -c.rank)


def rank(
    parent_text: str,
    pool: list[Candidate],
    texts: dict[str, str],
    limit: int,
    path_texts: list[str] | None = None,
    parent_categories: set[str] | None = None,
    candidate_categories: dict[str, set[str]] | None = None,
    backend: str = "tfidf",) -> list[Candidate]:
    usable = [c for c in pool if texts.get(c.title)]
    if not usable:
        return []

    path_texts = path_texts or []
    vectorizer = vectors.get(backend)

    corpus = [parent_text] + [texts[c.title] for c in usable] + path_texts
    matrix = vectorizer.encode(corpus)
    parent_vector = matrix[0]
    candidate_vectors = matrix[1 : 1 + len(usable)]
    path_vectors = matrix[1 + len(usable) :]
    reference = (
        path_vectors.mean(axis=0) if len(path_vectors) else candidate_vectors.mean(axis=0))

    relevance = candidate_vectors @ parent_vector
    band = _minmax(_bandedness(relevance))
    novelty = _minmax(1.0 - candidate_vectors @ reference)
    grounding = _minmax(
        np.array([
            category_affinity(
                (candidate_categories or {}).get(c.title, set()), parent_categories or set())
            for c in usable], dtype=np.float32))

    chosen: list[Candidate] = []
    chosen_vectors: list[np.ndarray] = []

    while usable and len(chosen) < limit:
        crowding = _minmax(
            np.array([
                max((float(vector @ picked) for picked in chosen_vectors), default=0.0)
                for vector in candidate_vectors], dtype=np.float32))

        best_index, best_score, best_parts = None, -np.inf, {}

        for index in range(len(usable)):
            parts = {
                "band": float(band[index]),
                "novelty": float(novelty[index]),
                "grounding": float(grounding[index]),
                "crowding": float(crowding[index]),}
            score = sum(
                WEIGHTS[key] * value * (-1 if key == "crowding" else 1)
                for key, value in parts.items())

            if score > best_score:
                best_index, best_score, best_parts = index, score, parts

        if best_index is None:
            break

        winner = usable[best_index]
        winner.score = float(best_score)
        winner.parts = best_parts | {"relevance": float(relevance[best_index])}
        chosen.append(winner)
        chosen_vectors.append(candidate_vectors[best_index])
        usable = [c for i, c in enumerate(usable) if i != best_index]
        candidate_vectors = np.delete(candidate_vectors, best_index, axis=0)
        relevance = np.delete(relevance, best_index)
        band = np.delete(band, best_index)
        novelty = np.delete(novelty, best_index)
        grounding = np.delete(grounding, best_index)

    return chosen


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values

    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.full(len(values), 0.5, dtype=np.float32)

    return ((values - low) / (high - low)).astype(np.float32)


def _bandedness(relevance: np.ndarray) -> np.ndarray:
    if len(relevance) < 2:
        return np.ones(len(relevance), dtype=np.float32)

    order = relevance.argsort().argsort().astype(np.float32)
    quantiles = order / (len(relevance) - 1)
    reach = max(BAND_QUANTILE, 1.0 - BAND_QUANTILE)

    return 1.0 - np.abs(quantiles - BAND_QUANTILE) / reach


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def category_affinity(left: set[str], right: set[str]) -> float:
    return _jaccard(_category_words(left), _category_words(right))


def _category_words(categories: set[str]) -> set[str]:
    words: set[str] = set()
    for category in categories:
        name = category.removeprefix("Category:").lower()
        words.update(
            word.rstrip("s") for word in re.findall(r"[a-z]{3,}", name)
            if word not in _CATEGORY_STOPWORDS)

    return words
