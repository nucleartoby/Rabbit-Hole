from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
from rabbithole import sources, vectors

SEARCH = "search"
CANDIDATES = "candidates"
RANKED = "ranked"
SESSION = "session"
TYPED = "typed"
KGE = "kge"
STAGES = (SEARCH, CANDIDATES, RANKED, SESSION, TYPED, KGE)
RANKING_STAGES = (RANKED, SESSION, TYPED, KGE)

BAND_QUANTILE = 0.62
BAND_FLOOR, BAND_CEILING = 0.25, 0.90

PROFILES = {
    RANKED: {
        "band": 0.30, "novelty": 0.15, "grounding": 0.35, "crowding": 0.20},
    SESSION: {
        "band": 0.25, "drift": 0.20, "novelty": 0.15, "grounding": 0.25, "crowding": 0.15},
    TYPED: {
        "band": 0.20, "drift": 0.15, "novelty": 0.10, "grounding": 0.20, "crowding": 0.10,
        "variety": 0.25},
    KGE: {
        "band": 0.18, "drift": 0.12, "novelty": 0.10, "grounding": 0.15, "crowding": 0.10,
        "variety": 0.20, "structural": 0.15},}

WEIGHTS = PROFILES[RANKED]
PENALTIES = frozenset({"crowding"})
UNKNOWN_VARIETY = 0.4

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


@dataclass
class Context:

    parent_text: str
    texts: dict[str, str]
    path_texts: list[str] = field(default_factory=list)
    parent_categories: set[str] = field(default_factory=set)
    candidate_categories: dict[str, set[str]] = field(default_factory=dict)
    families: dict[str, str] = field(default_factory=dict)
    structure: dict[str, float] = field(default_factory=dict)
    taste: float | None = None
    backend: str = "tfidf"


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


def band_centre(taste: float | None) -> float:
    if taste is None:
        return BAND_QUANTILE

    return min(max(float(taste), BAND_FLOOR), BAND_CEILING)


def rank(
    pool: list[Candidate],
    limit: int,
    context: Context,
    profile: str = RANKED,) -> list[Candidate]:
    weights = PROFILES[profile]
    usable = [c for c in pool if context.texts.get(c.title)]
    if not usable:
        return []

    matrix = vectors.get(context.backend).encode(
        [context.parent_text]
        + [context.texts[c.title] for c in usable]
        + list(context.path_texts))
    parent_vector = matrix[0]
    candidate_vectors = matrix[1 : 1 + len(usable)]
    visited = matrix[1 + len(usable) :]

    relevance = candidate_vectors @ parent_vector
    quantiles = _quantiles(relevance)
    static = {"band": _minmax(_bandedness(quantiles, band_centre(context.taste)))}

    if "grounding" in weights:
        static["grounding"] = _minmax(
            np.array([
                category_affinity(
                    context.candidate_categories.get(c.title, set()), context.parent_categories)
                for c in usable], dtype=np.float32))

    if "novelty" in weights:
        static["novelty"] = _minmax(
            _novelty(candidate_vectors, visited, legacy=profile == RANKED))

    if "drift" in weights:
        static["drift"] = _minmax(_drift(candidate_vectors, parent_vector, visited))

    if "structural" in weights:
        static["structural"] = _minmax(
            _structural([c.title for c in usable], context.structure))

    chosen: list[Candidate] = []
    chosen_vectors: list[np.ndarray] = []
    taken_families: dict[str, int] = {}
    remaining = list(range(len(usable)))

    while remaining and len(chosen) < limit:
        crowding = _minmax(
            np.array([
                max((float(candidate_vectors[index] @ picked) for picked in chosen_vectors),
                    default=0.0)
                for index in remaining], dtype=np.float32))
        variety = (
            _variety(usable, remaining, context.families, taken_families)
            if "variety" in weights else None)

        best_position, best_score, best_parts = None, -np.inf, {}

        for position, index in enumerate(remaining):
            parts = {name: float(values[index]) for name, values in static.items()}
            parts["crowding"] = float(crowding[position])
            if variety is not None:
                parts["variety"] = float(variety[position])

            score = sum(
                weights[name] * value * (-1 if name in PENALTIES else 1)
                for name, value in parts.items() if name in weights)

            if score > best_score:
                best_position, best_score, best_parts = position, score, parts

        if best_position is None:
            break

        winner_index = remaining[best_position]
        winner = usable[winner_index]
        winner.score = float(best_score)
        winner.parts = best_parts | {
            "relevance": float(relevance[winner_index]),
            "quantile": float(quantiles[winner_index]),}
        chosen.append(winner)
        chosen_vectors.append(candidate_vectors[winner_index])
        family = context.families.get(winner.title, "")
        if family:
            taken_families[family] = taken_families.get(family, 0) + 1
        remaining.pop(best_position)

    return chosen


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values

    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.full(len(values), 0.5, dtype=np.float32)

    return ((values - low) / (high - low)).astype(np.float32)


def _quantiles(relevance: np.ndarray) -> np.ndarray:
    if len(relevance) < 2:
        return np.full(len(relevance), 0.5, dtype=np.float32)

    order = relevance.argsort().argsort().astype(np.float32)

    return order / (len(relevance) - 1)


def _bandedness(quantiles: np.ndarray, centre: float = BAND_QUANTILE) -> np.ndarray:
    reach = max(centre, 1.0 - centre)

    return 1.0 - np.abs(quantiles - centre) / reach


def _novelty(candidate_vectors: np.ndarray, visited: np.ndarray, legacy: bool) -> np.ndarray:
    if len(visited) == 0:
        return 1.0 - candidate_vectors @ candidate_vectors.mean(axis=0)
    if legacy:
        return 1.0 - candidate_vectors @ visited.mean(axis=0)

    return 1.0 - (candidate_vectors @ visited.T).max(axis=1)


def _trajectory(visited: np.ndarray, parent_vector: np.ndarray) -> np.ndarray | None:
    if len(visited) >= 2:
        direction = visited[-1] - visited[0]
    elif len(visited) == 1:
        direction = parent_vector - visited[0]
    else:
        return None

    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None

    return direction / length


def _drift(
    candidate_vectors: np.ndarray, parent_vector: np.ndarray, visited: np.ndarray) -> np.ndarray:
    direction = _trajectory(visited, parent_vector)
    if direction is None:
        return np.full(len(candidate_vectors), 0.5, dtype=np.float32)

    return (candidate_vectors - parent_vector) @ direction


def _structural(titles: list[str], structure: dict[str, float]) -> np.ndarray:
    known = [structure[title] for title in titles if title in structure]
    if not known:
        return np.full(len(titles), 0.5, dtype=np.float32)

    fallback = float(np.mean(known))

    return np.array(
        [structure.get(title, fallback) for title in titles], dtype=np.float32)


def _variety(
    usable: list[Candidate],
    remaining: list[int],
    families: dict[str, str],
    taken: dict[str, int],) -> np.ndarray:
    values = []
    for index in remaining:
        family = families.get(usable[index].title, "")
        if not family:
            values.append(UNKNOWN_VARIETY)
        else:
            values.append(1.0 / (1.0 + taken.get(family, 0)))

    return np.array(values, dtype=np.float32)


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
