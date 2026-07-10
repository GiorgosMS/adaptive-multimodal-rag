import numpy as np
import pytest
from amrag.retrieve.rerank import BGEReranker, rerank
from amrag.types import Hit

TEXTS = {"d1": "irrelevant filler", "d2": "the answer is 42"}
HITS = [Hit("d1", 9.0, "passage", "text"), Hit("d2", 1.0, "passage", "text")]

class FakeScorer:
    def score(self, pairs): return [0.1 if "filler" in d else 0.9 for _, d in pairs]

class _TooFewScorer:
    def score(self, pairs): return [0.5]

class _TooManyScorer:
    def score(self, pairs): return [0.5] * (len(pairs) + 3)

class _StubModel:
    def __init__(self, ret): self.ret = ret
    def compute_score(self, pairs, normalize=True): return self.ret

def _stub_reranker(ret):
    r = BGEReranker.__new__(BGEReranker)
    r._m = _StubModel(ret)
    return r

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

def test_scorer_returning_fewer_scores_than_hits_raises():
    with pytest.raises(ValueError, match="3.*1|1.*3"):
        rerank("q", [Hit("d1", 1.0, "passage", "text")] * 3, TEXTS | {"d1": "x"}, _TooFewScorer(), k=3)

def test_scorer_returning_more_scores_than_hits_raises():
    with pytest.raises(ValueError, match="2.*5|5.*2"):
        rerank("q", [Hit("d1", 1.0, "passage", "text")] * 2, TEXTS | {"d1": "x"}, _TooManyScorer(), k=2)


class TestBGERerankerScore:
    def test_bare_python_float_single_pair(self):
        r = _stub_reranker(0.42)
        out = r.score([("q", "d")])
        assert out == [pytest.approx(0.42)]
        assert type(out[0]) is float

    def test_np_float32_scalar_single_pair(self):
        r = _stub_reranker(np.float32(0.9))
        out = r.score([("q", "d")])
        assert out == [pytest.approx(0.9, abs=1e-6)]
        assert type(out[0]) is float

    def test_np_float64_scalar_single_pair(self):
        r = _stub_reranker(np.float64(0.7))
        out = r.score([("q", "d")])
        assert out == [pytest.approx(0.7)]
        assert type(out[0]) is float

    def test_python_list_many_pairs(self):
        r = _stub_reranker([0.1, 0.2, 0.3])
        out = r.score([("q", "a"), ("q", "b"), ("q", "c")])
        assert out == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
        assert all(type(x) is float for x in out)

    def test_ndarray_float32_many_pairs(self):
        r = _stub_reranker(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        out = r.score([("q", "a"), ("q", "b"), ("q", "c")])
        assert len(out) == 3
        assert all(type(x) is float for x in out)
        assert out[0] == pytest.approx(0.1, abs=1e-6)
