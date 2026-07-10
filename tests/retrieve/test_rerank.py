import pytest
from amrag.retrieve.rerank import rerank
from amrag.types import Hit

TEXTS = {"d1": "irrelevant filler", "d2": "the answer is 42"}
HITS = [Hit("d1", 9.0, "passage", "text"), Hit("d2", 1.0, "passage", "text")]

class FakeScorer:
    def score(self, pairs): return [0.1 if "filler" in d else 0.9 for _, d in pairs]

def test_reranker_overrides_first_stage_order():
    out = rerank("what is the answer", HITS, TEXTS, FakeScorer(), k=2)
    assert [h.doc_id for h in out] == ["d2", "d1"]

def test_scores_are_replaced_by_reranker_scores():
    out = rerank("q", HITS, TEXTS, FakeScorer(), k=2)
    assert out[0].score == pytest.approx(0.9)

def test_granularity_and_modality_survive_reranking():
    out = rerank("q", HITS, TEXTS, FakeScorer(), k=1)
    assert (out[0].granularity, out[0].modality) == ("passage", "text")

def test_k_truncates_after_reordering():
    out = rerank("q", HITS, TEXTS, FakeScorer(), k=1)
    assert [h.doc_id for h in out] == ["d2"]

def test_hit_missing_from_doc_texts_raises_rather_than_scoring_empty_string():
    with pytest.raises(KeyError):
        rerank("q", [Hit("dX", 1.0, "passage", "text")], TEXTS, FakeScorer(), k=1)
