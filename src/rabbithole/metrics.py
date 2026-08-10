from __future__ import annotations
from dataclasses import dataclass, field
from statistics import fmean
import numpy as np
from rabbithole import relate, vectors

HIGHER_IS_BETTER = {
    "grounding": True,
    "link_grounding": True,
    "band_rate": True,
    "yield_rate": True,
    "lexical_leak": False,
    "junk_rate": False,
    "redundancy": False,}

QUALITY_WEIGHTS = {
    "lexical_leak": 0.25,
    "redundancy": 0.20,
    "grounding": 0.15,
    "band_rate": 0.15,
    "junk_rate": 0.10,
    "link_grounding": 0.10,
    "yield_rate": 0.05,}



@dataclass
class Expansion:

    parent: str
    parent_text: str
    picks: list[str]
    texts: dict[str, str] = field(default_factory=dict)
    requested: int = 8
    parent_categories: set[str] = field(default_factory=set)
    pick_categories: dict[str, set[str]] = field(default_factory=dict)
    parent_links: set[str] = field(default_factory=set)


def lexical_leak(expansion: Expansion) -> float:
    if not expansion.picks:
        return 0.0

    parent_tokens = relate.title_tokens(expansion.parent)
    leaked = sum(
        1 for pick in expansion.picks if relate.title_tokens(pick) & parent_tokens)

    return leaked / len(expansion.picks)


def junk_rate(expansion: Expansion) -> float:
    if not expansion.picks:
        return 0.0

    return sum(1 for pick in expansion.picks if relate.is_junk(pick)) / len(expansion.picks)


def grounding(expansion: Expansion) -> float:
    if not expansion.picks or not expansion.parent_categories:
        return 0.0

    return fmean(
        _jaccard(expansion.pick_categories.get(pick, set()), expansion.parent_categories)
        for pick in expansion.picks)


def link_grounding(expansion: Expansion) -> float:
    if not expansion.picks or not expansion.parent_links:
        return 0.0

    return sum(1 for pick in expansion.picks if pick in expansion.parent_links) / len(
        expansion.picks)


def yield_rate(expansion: Expansion) -> float:
    if expansion.requested <= 0:
        return 0.0

    return min(len(expansion.picks) / expansion.requested, 1.0)


def _vector_metrics(expansion: Expansion, backend: str) -> dict[str, float]:
    usable = [pick for pick in expansion.picks if expansion.texts.get(pick)]
    if len(usable) < 1 or not expansion.parent_text:
        return {"redundancy": 0.0, "band_rate": 0.0}

    matrix = vectors.get(backend).encode(
        [expansion.parent_text] + [expansion.texts[pick] for pick in usable])
    parent_vector, pick_vectors = matrix[0], matrix[1:]

    low, high = vectors.band_for(backend)
    to_parent = pick_vectors @ parent_vector
    band_rate = float(np.mean((to_parent >= low) & (to_parent <= high)))

    if len(usable) < 2:
        return {"redundancy": 0.0, "band_rate": band_rate}

    pairwise = vectors.cosine_matrix(pick_vectors)
    upper = pairwise[np.triu_indices(len(usable), k=1)]

    return {"redundancy": float(np.mean(upper)), "band_rate": band_rate}


def score(expansion: Expansion, backend: str = "tfidf") -> dict[str, float]:
    return {
        "lexical_leak": lexical_leak(expansion),
        "junk_rate": junk_rate(expansion),
        "grounding": grounding(expansion),
        "link_grounding": link_grounding(expansion),
        "yield_rate": yield_rate(expansion),
        **_vector_metrics(expansion, backend),}


def quality(scored: dict[str, float]) -> float:
    total = 0.0
    for name, weight in QUALITY_WEIGHTS.items():
        value = scored.get(name, 0.0)
        total += weight * (value if HIGHER_IS_BETTER[name] else 1.0 - value)

    return total


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}

    summary = {name: fmean(row.get(name, 0.0) for row in rows) for name in HIGHER_IS_BETTER}
    summary["quality"] = quality(summary)

    return summary


def drift(root_text: str, depth_texts: dict[int, list[str]], backend: str = "tfidf"):
    flat = [(depth, text) for depth, texts in depth_texts.items() for text in texts if text]
    if not flat or not root_text:
        return {}

    matrix = vectors.get(backend).encode([root_text] + [text for _, text in flat])
    to_root = matrix[1:] @ matrix[0]

    by_depth: dict[int, list[float]] = {}
    for (depth, _), similarity in zip(flat, to_root, strict=True):
        by_depth.setdefault(depth, []).append(float(similarity))

    return {depth: fmean(values) for depth, values in sorted(by_depth.items())}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)
