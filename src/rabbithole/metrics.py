from __future__ import annotations
from dataclasses import dataclass, field
from statistics import fmean
import numpy as np
from rabbithole import labels, relate, vectors

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

MINIMUM_LABELS = 30
TRUST_THRESHOLD = 0.5


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


def oriented(scored: dict[str, float]) -> dict[str, float]:
    return {
        name: (scored.get(name, 0.0) if higher else 1.0 - scored.get(name, 0.0))
        for name, higher in HIGHER_IS_BETTER.items()}


def quality(scored: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or QUALITY_WEIGHTS
    features = oriented(scored)

    return sum(weight * features[name] for name, weight in weights.items())


def aggregate(
    rows: list[dict[str, float]], weights: dict[str, float] | None = None) -> dict[str, float]:
    if not rows:
        return {}

    summary = {name: fmean(row.get(name, 0.0) for row in rows) for name in HIGHER_IS_BETTER}
    summary["quality"] = quality(summary, weights)

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


@dataclass(frozen=True)
class Calibration:

    weights: dict[str, float]
    agreement: float            # spearman fitted quality against human rating
    baseline: float             # spearman hand-set weights against human rating
    samples: int

    @property
    def trusted(self) -> bool:
        return self.samples >= MINIMUM_LABELS and self.agreement >= TRUST_THRESHOLD

    def verdict(self) -> str:
        if self.samples < MINIMUM_LABELS:
            return (
                f"uncalibrated: {self.samples}/{MINIMUM_LABELS} judgements. "
                "quality is a proxy nobody has checked; rank stages by it at your own risk")
        if self.agreement < TRUST_THRESHOLD:
            return (
                f"calibrated on {self.samples} judgements but agreement is only "
                f"{self.agreement:.2f}; the metrics do not capture what you are rating")

        return (
            f"calibrated on {self.samples} judgements, agreement {self.agreement:.2f} "
            f"(hand-set weights scored {self.baseline:.2f})")


def _ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()

    return ranks


def spearman(left, right) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(left) != len(right):
        return 0.0

    a, b = _ranks(left), _ranks(right)
    a, b = a - a.mean(), b - b.mean()
    spread = float(np.linalg.norm(a) * np.linalg.norm(b))
    if spread < 1e-12:
        return 0.0

    return float(a @ b / spread)


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    descending = np.sort(vector)[::-1]
    cumulative = np.cumsum(descending)
    indices = np.arange(1, len(vector) + 1)
    supported = descending - (cumulative - 1.0) / indices > 0
    rho = int(indices[supported][-1])
    theta = (cumulative[rho - 1] - 1.0) / rho

    return np.maximum(vector - theta, 0.0)


def fit_weights(
    matrix: np.ndarray, ratings: np.ndarray, steps: int = 4000, rate: float = 0.05) -> np.ndarray:
    weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])

    for _ in range(steps):
        gradient = 2.0 * matrix.T @ (matrix @ weights - ratings) / len(ratings)
        weights = _project_simplex(weights - rate * gradient)

    return weights


def calibrate(judgements: list[labels.Judgement]) -> Calibration:
    usable = [judgement for judgement in judgements if judgement.scored]
    names = list(HIGHER_IS_BETTER)
    if len(usable) < 2:
        return Calibration(dict(QUALITY_WEIGHTS), 0.0, 0.0, len(usable))

    matrix = np.array(
        [[oriented(judgement.scored)[name] for name in names] for judgement in usable])
    ratings = np.array([judgement.rating for judgement in usable])
    fitted = fit_weights(matrix, ratings)
    hand_set = np.array([QUALITY_WEIGHTS[name] for name in names])

    return Calibration(
        weights={name: float(weight) for name, weight in zip(names, fitted, strict=True)},
        agreement=spearman(matrix @ fitted, ratings),
        baseline=spearman(matrix @ hand_set, ratings),
        samples=len(usable),)
