import pytest
from amrag.eval.retrieval import recall_at_k, precision_at_k, ndcg_at_k, mrr

QRELS = {"d1": 1, "d2": 1}
RUN = ["d3", "d1", "d2"]

def test_recall_at_k():
    assert recall_at_k(RUN, QRELS, 3) == pytest.approx(1.0)
    assert recall_at_k(RUN, QRELS, 2) == pytest.approx(0.5)

def test_precision_at_k():
    assert precision_at_k(RUN, QRELS, 3) == pytest.approx(2 / 3)

def test_ndcg_at_k():
    # DCG@3  = 0/log2(2) + 1/log2(3) + 1/log2(4) = 1.130930
    # IDCG@3 = 1/log2(2) + 1/log2(3)             = 1.630930
    assert ndcg_at_k(RUN, QRELS, 3) == pytest.approx(0.693426, abs=1e-6)

def test_mrr_uses_first_relevant_rank():
    assert mrr(RUN, QRELS) == pytest.approx(0.5)

def test_no_relevant_docs_scores_zero():
    assert ndcg_at_k(["x"], QRELS, 3) == 0.0
    assert mrr(["x"], QRELS) == 0.0
    assert recall_at_k(["x"], QRELS, 3) == 0.0

def test_empty_qrels_does_not_divide_by_zero():
    assert recall_at_k(RUN, {}, 3) == 0.0
    assert ndcg_at_k(RUN, {}, 3) == 0.0

def test_ndcg_matches_pytrec_eval():
    """Our nDCG must agree with the reference IR implementation.

    Valid ONLY for binary relevance: trec_eval uses exponential gain (2^rel - 1),
    ours uses linear gain (rel). For rel in {0,1} these coincide (2^1-1 == 1).
    If graded relevance is ever introduced, this test will break -- correctly.
    """
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator({"q1": QRELS}, {"ndcg_cut_3"})
    scored = ev.evaluate({"q1": {d: float(len(RUN) - i) for i, d in enumerate(RUN)}})
    assert ndcg_at_k(RUN, QRELS, 3) == pytest.approx(scored["q1"]["ndcg_cut_3"], abs=1e-6)

def test_ndcg_idcg_truncation_branch():
    """Regression: nDCG IDCG must be truncated to k, not computed over all relevant docs.

    This case has |relevant| > k, forcing the IDCG[:k] truncation branch that
    test_ndcg_matches_pytrec_eval cannot reach (|relevant| = 2 <= k = 3).
    Without this truncation, a buggy implementation would compute IDCG over all
    4 relevant docs and return 0.390380 instead of the correct 0.613147.
    """
    import pytrec_eval
    qrels = {"d1": 1, "d2": 1, "d3": 1, "d4": 1}  # 4 relevant
    run = ["d1", "d5", "d6", "d7"]                 # only d1 retrieved, at rank 1
    k = 2

    # Correct value with IDCG truncated to k=2
    assert ndcg_at_k(run, qrels, k) == pytest.approx(0.613147, abs=1e-6)

    # Cross-check against pytrec_eval reference
    ev = pytrec_eval.RelevanceEvaluator({"q1": qrels}, {"ndcg_cut_2"})
    scored = ev.evaluate({"q1": {d: float(len(run) - i) for i, d in enumerate(run)}})
    assert ndcg_at_k(run, qrels, k) == pytest.approx(scored["q1"]["ndcg_cut_2"], abs=1e-6)

def test_all_qrels_are_binary():
    """Guards the assumption the pytrec_eval cross-check depends on."""
    assert set(QRELS.values()) <= {0, 1}
