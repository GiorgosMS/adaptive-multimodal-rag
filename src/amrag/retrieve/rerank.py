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
        from FlagEmbedding import FlagReranker
        self._m = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device=device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw = self._m.compute_score(pairs, normalize=True)
        # compute_score returns a bare scalar for one pair, a sequence for many.
        # np.float32 is not a Python float and is not iterable, so neither
        # isinstance(raw, float) nor list(raw) is safe on its own.
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
