"""Cross-encoder reranking (deck slide 28): score (query, chunk) pairs jointly
rather than trusting bi-encoder cosine similarity.
"""
from typing import Protocol

import numpy as np

from amrag.types import Hit


class Scorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


_MAX_LENGTH = 512
_BATCH_SIZE = 16


class BGEReranker:
    """Cross-encoder scorer with a bounded memory footprint.

    Both bounds are load-bearing, not tuning knobs. BAAI/bge-reranker-v2-m3
    declares `model_max_length = 8192`, and CrossEncoder pads every batch to
    its longest member. QASPER's longest paragraph is 5,303 tokens, so the
    library defaults (unbounded length, batch 32) tried to allocate 3.37 GiB
    of attention for a single batch and hit CUDA OOM on a 12 GB card. It
    "worked" on CPU only because the same allocation came out of system RAM,
    at roughly 150x the wall-clock cost.

    max_length=512 is near-lossless on this corpus: 52 of 20,221 QASPER
    paragraphs (0.26%) exceed 512 tokens; p95 is 275 and p99 is 392. The
    truncated tail is 2 paragraphs beyond 1024 tokens. Raise both bounds via
    the constructor if you have the VRAM and a longer-passage corpus.
    """

    def __init__(self, device: str = "cuda", max_length: int = _MAX_LENGTH,
                 batch_size: int = _BATCH_SIZE) -> None:
        # Lazy import: FlagEmbedding's compute_score calls
        # tokenizer.prepare_for_model, removed in transformers 5.x. This
        # backend (sentence_transformers.CrossEncoder) works on 5.13.0.
        # Import the module, not the symbol, so tests can monkeypatch it.
        import sentence_transformers
        self._batch_size = batch_size
        self._m = sentence_transformers.CrossEncoder(
            "BAAI/bge-reranker-v2-m3", device=device, max_length=max_length,
        )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw = self._m.predict(pairs, batch_size=self._batch_size)
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
