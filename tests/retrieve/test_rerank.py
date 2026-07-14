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
    """Stubs sentence_transformers.CrossEncoder's `.predict(pairs)`."""
    def __init__(self, ret): self.ret = ret
    def predict(self, pairs, **kwargs): return self.ret

def _stub_reranker(ret):
    # __new__ bypasses __init__, so every attribute __init__ would have set
    # must be set here too, or these tests fail for reasons unrelated to the
    # return-type coercion they exist to pin.
    r = BGEReranker.__new__(BGEReranker)
    r._m = _StubModel(ret)
    r._batch_size = 16
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

    def test_ndarray_float32_single_pair(self):
        """The real backend, CrossEncoder.predict, always returns a
        np.ndarray -- shape (1,) for one pair, never a bare scalar. This is
        the exact shape verified live: array([0.98785], dtype=float32)."""
        r = _stub_reranker(np.array([0.98785], dtype=np.float32))
        out = r.score([("q", "d")])
        assert out == [pytest.approx(0.98785, abs=1e-5)]
        assert type(out[0]) is float


class _RecordingCrossEncoder:
    """Captures the kwargs BGEReranker hands to CrossEncoder, without weights."""
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.init_kwargs = kwargs
        self.predict_kwargs = None

    def predict(self, pairs, **kwargs):
        self.predict_kwargs = kwargs
        return np.zeros(len(pairs), dtype=np.float32)


@pytest.fixture
def recording_cross_encoder(monkeypatch):
    import sentence_transformers
    created = []

    def factory(*a, **kw):
        m = _RecordingCrossEncoder(*a, **kw)
        created.append(m)
        return m

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", factory)
    return created


class TestBGERerankerBoundsMemory:
    """BAAI/bge-reranker-v2-m3 declares model_max_length = 8192, and
    CrossEncoder pads every batch to its longest member. QASPER's longest
    paragraph is 5,303 tokens, so an unbounded batch of 32 tried to allocate
    3.37 GiB of attention and OOM'd a 12 GB GPU. Only 52 of 20,221 QASPER
    paragraphs (0.26%) exceed 512 tokens; p99 is 392. Bounding at 512 is
    therefore near-lossless, and is a correctness contract, not a tuning knob:
    without it the +rerank rung cannot run on a consumer GPU at all.
    """

    def test_sequence_length_is_bounded_at_construction(self, recording_cross_encoder):
        BGEReranker(device="cpu")
        (model,) = recording_cross_encoder
        assert model.init_kwargs.get("max_length") == 512, (
            "max_length must be pinned; inheriting the model's 8192 default "
            "lets one long paragraph OOM the whole batch"
        )

    def test_batch_size_is_bounded_at_predict(self, recording_cross_encoder):
        r = BGEReranker(device="cpu")
        r.score([("q", "d")] * 4)
        (model,) = recording_cross_encoder
        assert model.predict_kwargs.get("batch_size") == 16

    def test_bounds_are_overridable_for_a_larger_gpu(self, recording_cross_encoder):
        r = BGEReranker(device="cpu", max_length=1024, batch_size=64)
        r.score([("q", "d")])
        (model,) = recording_cross_encoder
        assert model.init_kwargs.get("max_length") == 1024
        assert model.predict_kwargs.get("batch_size") == 64

    def test_device_is_still_forwarded(self, recording_cross_encoder):
        BGEReranker(device="cpu")
        (model,) = recording_cross_encoder
        assert model.init_kwargs.get("device") == "cpu"


@pytest.mark.slow
def test_bge_reranker_ranks_relevant_document_above_irrelevant():
    """The test that would have caught the transformers-5.x crash: loads the
    real BAAI/bge-reranker-v2-m3 weights via CrossEncoder and exercises
    BGEReranker.score end to end. CPU only (see task constraints). Assert on
    ORDERING, not absolute score values -- those are not a stable contract.
    """
    r = BGEReranker(device="cpu")
    out = r.score([
        ("what is the capital of France?", "Paris is the capital of France."),
        ("what is the capital of France?", "Bananas are a good source of potassium."),
    ])
    assert len(out) == 2
    assert all(type(x) is float for x in out)
    assert out[0] > out[1]
