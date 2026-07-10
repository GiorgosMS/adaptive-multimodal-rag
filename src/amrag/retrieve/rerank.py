"""Cross-encoder reranking (deck slide 28): score (query, chunk) pairs jointly
rather than trusting bi-encoder cosine similarity.
"""
from typing import Protocol

import numpy as np

from amrag.types import Hit


class Scorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class BGEReranker:
    def __init__(self, device: str = "cuda") -> None:
        # Lazy import: FlagEmbedding's compute_score calls
        # tokenizer.prepare_for_model, removed in transformers 5.x. This
        # backend (sentence_transformers.CrossEncoder) works on 5.13.0.
        from sentence_transformers import CrossEncoder
        self._m = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw = self._m.predict(pairs)
        # predict() returns a np.ndarray of np.float32, even for a single
        # pair (shape (1,)). np.float32 is not a Python float and a bare
        # np.float32 scalar is not iterable, so neither isinstance(raw, float)
        # nor list(raw) is safe on its own; np.atleast_1d is correct for an
        # ndarray of any shape too, so this also covers larger batches.
        return [float(x) for x in np.atleast_1d(np.asarray(raw, dtype=float))]


def rerank(query: str, hits: list[Hit], doc_texts: dict[str, str],
           scorer: Scorer, k: int) -> list[Hit]:
    if not hits:
        return []
    pairs = [(query, doc_texts[h.doc_id]) for h in hits]   # KeyError is intentional
    scores = scorer.score(pairs)
    if len(scores) != len(hits):
        raise ValueError(
            f"Scorer returned {len(scores)} scores for {len(hits)} hits. "
            f"rerank() cannot align them; a silent zip() truncation would "
            f"drop candidates and corrupt retrieval metrics."
        )
    rescored = [
        Hit(doc_id=h.doc_id, score=float(s), granularity=h.granularity,
            modality=h.modality, page=h.page, span=h.span)
        for h, s in zip(hits, scores)
    ]
    rescored.sort(key=lambda h: -h.score)
    return rescored[:k]
