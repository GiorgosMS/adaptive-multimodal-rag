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

@pytest.mark.slow
def test_bge_m3_encoder_produces_normalised_vectors():
    from amrag.index.text import BGEM3Encoder
    v = BGEM3Encoder().encode(["hello world"])
    assert v.shape[1] == 1024
    assert np.linalg.norm(v[0]) == pytest.approx(1.0, abs=1e-3)
