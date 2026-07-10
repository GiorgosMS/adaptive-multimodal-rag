import numpy as np
import pytest
from amrag.corpus.base import Document
from amrag.index.text import DenseRetriever

DOCS = [Document("d1", "cat", {}), Document("d2", "photon", {})]

class FakeEncoder:
    """d1 -> [1,0], d2 -> [0,1]; query 'cat' -> [1,0]."""
    TABLE = {"cat": [1.0, 0.0], "photon": [0.0, 1.0]}
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.TABLE.get(t, [0.0, 0.0]) for t in texts], dtype=np.float32)

def test_retrieves_nearest_neighbour_by_cosine():
    r = DenseRetriever.build(DOCS, FakeEncoder())
    hits = r.retrieve("cat", k=1)
    assert hits[0].doc_id == "d1"
    assert hits[0].score == pytest.approx(1.0)

def test_declares_passage_granularity_and_text_modality():
    r = DenseRetriever.build(DOCS, FakeEncoder())
    h = r.retrieve("cat", k=1)[0]
    assert (h.granularity, h.modality) == ("passage", "text")

def test_orders_all_docs_when_k_exceeds_corpus():
    r = DenseRetriever.build(DOCS, FakeEncoder())
    hits = r.retrieve("cat", k=10)
    assert [h.doc_id for h in hits] == ["d1", "d2"]


class UnnormalisedDocEncoder:
    """d1 -> unit vector, d2 -> norm ~4.24 (not unit). Reproduces the
    reviewer's counter-example: without the guard, d2 (dot=3.0) would
    outrank d1 (dot=1.0, true cosine 1.0)."""
    TABLE = {"cat": [1.0, 0.0], "dog": [3.0, 3.0]}
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.TABLE.get(t, [0.0, 0.0]) for t in texts], dtype=np.float32)

def test_build_rejects_unnormalised_document_vectors():
    docs = [Document("d1", "cat", {}), Document("d2", "dog", {})]
    with pytest.raises(ValueError, match=r"Encoder\.encode\(documents\)"):
        DenseRetriever.build(docs, UnnormalisedDocEncoder())


class UnnormalisedQueryEncoder:
    """Documents encode to unit vectors; the query text encodes to a
    non-unit vector, isolating the guard's query-side check from its
    document-side check."""
    DOC_TABLE = {"cat": [1.0, 0.0], "photon": [0.0, 1.0]}
    def encode(self, texts: list[str]) -> np.ndarray:
        if texts == ["bad query"]:
            return np.array([[2.0, 0.0]], dtype=np.float32)
        return np.array([self.DOC_TABLE[t] for t in texts], dtype=np.float32)

def test_retrieve_rejects_unnormalised_query_vector():
    r = DenseRetriever.build(DOCS, UnnormalisedQueryEncoder())
    with pytest.raises(ValueError, match=r"Encoder\.encode\(query\)"):
        r.retrieve("bad query", k=1)


class GroupedTieEncoder:
    """26 two-dimensional unit vectors forming 5 cosine-similarity tie
    groups against a fixed query vector [1, 0]. Empirically verified: with
    these exact scores, numpy's default (quicksort) argsort reorders
    within-group ties relative to corpus order, while kind='stable' does
    not -- so this fixture actually exercises the bug, unlike an all-equal-
    score array (which happens not to trigger reordering at this size)."""
    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t == "__query__":
                out.append([1.0, 0.0])
            else:
                s = (int(t) // 5) * 0.15
                out.append([s, (1.0 - s ** 2) ** 0.5])
        return np.array(out, dtype=np.float32)

def test_ties_are_broken_by_stable_corpus_order():
    docs = [Document(f"d{i}", str(i), {}) for i in range(26)]
    r = DenseRetriever.build(docs, GroupedTieEncoder())
    expected = [f"d{i}" for i in [
        25, 20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14,
        5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
    ]]
    hits = r.retrieve("__query__", k=26)
    assert [h.doc_id for h in hits] == expected
    # repeated calls must be deterministic, not just "happens to match once"
    hits_again = r.retrieve("__query__", k=26)
    assert [h.doc_id for h in hits_again] == expected

@pytest.mark.slow
def test_bge_m3_encoder_produces_normalised_vectors():
    from amrag.index.text import BGEM3Encoder
    v = BGEM3Encoder().encode(["hello world"])
    assert v.shape[1] == 1024
    assert np.linalg.norm(v[0]) == pytest.approx(1.0, abs=1e-3)
