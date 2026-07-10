"""Text retrievers: BM25 (lexical) and dense (semantic).

Deck slide 25: dense captures paraphrase, sparse captures exact terms, IDs and
rare phrases. On a paper corpus the sparse arm is what finds "ColBERT"; the
dense arm is what finds "late interaction retrieval". (Symbol-bearing phrases
like "nDCG@10" are split by `_tokenize` into ["ndcg", "10"] -- see its
docstring for why that tradeoff is the measured-better one.)
"""
import re
from typing import Iterable, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from amrag.corpus.base import Document
from amrag.types import Hit


_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, then split on non-word characters.

    `text.lower().split()` welds punctuation onto tokens: a document's "BERT,"
    never matches a query's "BERT", and a question ending "...F1 score?" looks
    for the token "score?", which occurs nowhere. Measured standalone on QASPER
    (20,221 paragraphs, 1,451 queries), the fix lifts BM25's Recall@10 from
    0.0997 to 0.1446 -- +45% relative. Whether that is enough for the fused
    +hybrid rung to beat the dense-only naive rung is a separate question, to
    be measured rather than assumed: RRF interleaves the two arms one-for-one,
    so a still-weaker sparse arm can drag the fusion down even after this fix.

    The tradeoff: `\\w+` also splits "nDCG@10" into ["ndcg", "10"], so an exact
    symbol-bearing phrase is no longer a single token. That costs less than the
    punctuation welding it removes; the numbers above are the evidence.
    """
    return _WORD.findall(text.lower())


class BM25Retriever:
    def __init__(self, doc_ids: list[str], bm25: BM25Okapi) -> None:
        self._doc_ids = doc_ids
        self._bm25 = bm25

    @classmethod
    def build(cls, docs: Iterable[Document]) -> "BM25Retriever":
        docs = list(docs)
        if not docs:
            raise ValueError("cannot build BM25 index over zero documents")
        return cls([d.doc_id for d in docs], BM25Okapi([_tokenize(d.text) for d in docs]))

    def retrieve(self, query: str, k: int) -> list[Hit]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Hit(doc_id=self._doc_ids[i], score=float(scores[i]),
                granularity="passage", modality="text")
            for i in order
        ]


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised embeddings."""
        ...


_NORM_TOL = 1e-3


def _assert_l2_normalised(vectors: np.ndarray, source: str) -> None:
    """Encoders must return unit vectors; dot product is only cosine if they do."""
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_TOL):
        bad = float(norms[np.argmax(np.abs(norms - 1.0))])
        raise ValueError(
            f"{source} returned vectors that are not L2-normalised "
            f"(worst norm {bad:.6f}, tolerance {_NORM_TOL}). "
            f"DenseRetriever treats the dot product as cosine similarity, "
            f"which is only valid for unit vectors."
        )


class BGEM3Encoder:
    """BAAI/bge-m3 dense head. 1024-dim, L2-normalised.

    No onnx/ directory in the snapshot: `sentence-transformers` pulls both
    `model.safetensors` (2.2 GB) and the legacy `pytorch_model.bin` (2.2 GB)
    -- the same weights in two formats, ~4.3 GB total on disk. If disk ever
    gets tight, `snapshot_download(..., ignore_patterns=["pytorch_model.bin"])`
    would halve that.
    """
    def __init__(self, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer("BAAI/bge-m3", device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._m.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=16, show_progress_bar=False,
        ).astype(np.float32)


class DenseRetriever:
    def __init__(self, doc_ids: list[str], matrix: np.ndarray, encoder: Encoder) -> None:
        self._doc_ids = doc_ids
        self._matrix = matrix          # (n, dim), L2-normalised
        self._encoder = encoder

    @classmethod
    def build(cls, docs: Iterable[Document], encoder: Encoder) -> "DenseRetriever":
        docs = list(docs)
        if not docs:
            raise ValueError("cannot build dense index over zero documents")
        matrix = encoder.encode([d.text for d in docs])
        _assert_l2_normalised(matrix, source="Encoder.encode(documents)")
        return cls([d.doc_id for d in docs], matrix, encoder)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        q_matrix = self._encoder.encode([query])
        _assert_l2_normalised(q_matrix, source="Encoder.encode(query)")
        q = q_matrix[0]
        scores = self._matrix @ q      # cosine, inputs are normalised
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(doc_id=self._doc_ids[i], score=float(scores[i]),
                granularity="passage", modality="text")
            for i in order
        ]
