from __future__ import annotations
import importlib.util
import math
import re
import numpy as np

_TOKEN = re.compile(r"[a-z][a-z'-]+")

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "in", "into", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "them", "there", "these", "they",
    "this", "to", "was", "were", "which", "who", "will", "with", "would", "you",
    "your", "not", "no", "than", "then", "when", "where", "while", "also", "may",
    "can", "could"})


def _tokenise(text: str) -> list[str]:
    return [
        token for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2]


class TfidfVectorizer:

    name = "tfidf"

    def encode(self, texts: list[str]) -> np.ndarray:
        docs = [_tokenise(text) for text in texts]
        vocabulary = {term: i for i, term in enumerate(sorted({t for d in docs for t in d}))}
        matrix = np.zeros((len(docs), len(vocabulary)), dtype=np.float32)
        if not vocabulary:
            return matrix

        for row, doc in enumerate(docs):
            for term in doc:
                matrix[row, vocabulary[term]] += 1.0

        document_frequency = (matrix > 0).sum(axis=0)
        idf = np.log((1.0 + len(docs)) / (1.0 + document_frequency)) + 1.0
        matrix = np.log1p(matrix) * idf

        return _normalise(matrix)


class MiniLmVectorizer:

    name = "minilm"

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_id = model_id
        self._loaded = None

    def _model(self):
        if self._loaded is None:
            from sentence_transformers import SentenceTransformer

            self._loaded = SentenceTransformer(self._model_id)
        return self._loaded

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        return _normalise(
            np.asarray(self._model().encode(texts, show_progress_bar=False), dtype=np.float32))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


BANDS = {"tfidf": (0.08, 0.20), "minilm": (0.35, 0.65)}


def band_for(name: str) -> tuple[float, float]:
    return BANDS.get(name, BANDS["tfidf"])


def available() -> list[str]:
    names = ["tfidf"]
    if importlib.util.find_spec("sentence_transformers"):
        names.append("minilm")
    return names


def get(name: str = "tfidf"):
    if name == "minilm":
        if "minilm" not in available():
            raise RuntimeError(
                "the 'minilm' backend needs sentence-transformers: pip install -e '.[nlp]'")
        return MiniLmVectorizer()

    if name != "tfidf":
        raise ValueError(f"unknown vectorizer {name!r}; available: {available()}")

    return TfidfVectorizer()


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix @ matrix.T


def entropy_of(values: list[float]) -> float:
    total = sum(values)
    if total <= 0.0:
        return 0.0

    return -sum((v / total) * math.log(v / total) for v in values if v > 0.0)
