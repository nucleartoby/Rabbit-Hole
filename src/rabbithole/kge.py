from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from rabbithole import config, wikidata


# h + r ~= t
@dataclass(frozen=True)
class Structure:

    by_title: dict[str, np.ndarray] = field(default_factory=dict)
    parent: np.ndarray | None = None

    def similarity(self, title: str) -> float | None:
        vector = self.by_title.get(title)
        if vector is None or self.parent is None:
            return None

        return float(vector @ self.parent)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def train(
    triples: list[tuple[str, str, str]],
    dim: int = 32,
    epochs: int = 120,
    margin: float = 1.0,
    rate: float = 0.05,
    seed: int = 0,) -> dict[str, np.ndarray]:
    entities = sorted({node for head, _, tail in triples for node in (head, tail)})
    relations = sorted({relation for _, relation, _ in triples})
    if dim < 2 or len(entities) < 2 or not relations:
        return {}

    entity_index = {name: position for position, name in enumerate(entities)}
    relation_index = {name: position for position, name in enumerate(relations)}
    rng = np.random.default_rng(seed)
    bound = 6.0 / math.sqrt(dim)

    embeddings = rng.uniform(-bound, bound, (len(entities), dim)).astype(np.float32)
    translations = _unit(
        rng.uniform(-bound, bound, (len(relations), dim)).astype(np.float32))

    heads = np.array([entity_index[head] for head, _, _ in triples])
    kinds = np.array([relation_index[relation] for _, relation, _ in triples])
    tails = np.array([entity_index[tail] for _, _, tail in triples])

    for _ in range(epochs):
        embeddings = _unit(embeddings)
        corrupt_head = rng.random(len(triples)) < 0.5
        noise = rng.integers(0, len(entities), len(triples))
        negative_heads = np.where(corrupt_head, noise, heads)
        negative_tails = np.where(corrupt_head, tails, noise)

        positive = embeddings[heads] + translations[kinds] - embeddings[tails]
        negative = embeddings[negative_heads] + translations[kinds] - embeddings[negative_tails]
        near = np.linalg.norm(positive, axis=1)
        far = np.linalg.norm(negative, axis=1)
        active = (margin + near - far) > 0.0
        if not active.any():
            break

        toward = positive[active] / np.maximum(near[active], 1e-9)[:, None]
        away = negative[active] / np.maximum(far[active], 1e-9)[:, None]

        entity_gradient = np.zeros_like(embeddings)
        relation_gradient = np.zeros_like(translations)
        np.add.at(entity_gradient, heads[active], toward)
        np.add.at(entity_gradient, tails[active], -toward)
        np.add.at(entity_gradient, negative_heads[active], -away)
        np.add.at(entity_gradient, negative_tails[active], away)
        np.add.at(relation_gradient, kinds[active], toward - away)

        embeddings -= rate * entity_gradient
        translations -= rate * relation_gradient

    return dict(zip(entities, _unit(embeddings), strict=True))


def embed(graph: wikidata.Graph, dim: int | None = None, epochs: int | None = None) -> Structure:
    by_qid = train(
        graph.triples,
        dim=config.KGE_DIM if dim is None else dim,
        epochs=config.KGE_EPOCHS if epochs is None else epochs,)
    if not by_qid:
        return Structure()

    return Structure(
        by_title={
            title: by_qid[qid] for title, qid in graph.qids.items() if qid in by_qid},
        parent=by_qid.get(graph.parent_qid),)
