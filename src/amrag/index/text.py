"""Text retrievers: BM25 (lexical) and dense (semantic).

Deck slide 25: dense captures paraphrase, sparse captures exact terms, IDs and
rare phrases. On a paper corpus the sparse arm is what finds "ColBERT" or
"nDCG@10"; the dense arm is what finds "late interaction retrieval".
"""
from typing import Iterable

from rank_bm25 import BM25Okapi

from amrag.corpus.base import Document
from amrag.types import Hit


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


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
