import pytest
from amrag.eval.answer import score_answers

GOLD = {
    "q1": {"answers": ["backpropagation"], "evidence": ["Nets are trained with backprop."]},
}

def test_exact_match_answer_scores_one():
    s = score_answers({"q1": "backpropagation"}, GOLD)
    assert s["answer_f1"] == pytest.approx(1.0)

def test_completely_wrong_answer_scores_zero():
    s = score_answers({"q1": "photons"}, GOLD)
    assert s["answer_f1"] == pytest.approx(0.0)

def test_partial_token_overlap_scores_between_zero_and_one():
    s = score_answers({"q1": "backpropagation and photons"}, GOLD)
    assert 0.0 < s["answer_f1"] < 1.0

def test_missing_prediction_is_scored_as_empty_not_skipped():
    """A question we failed to answer must count against us, not vanish."""
    s = score_answers({}, GOLD)
    assert s["answer_f1"] == pytest.approx(0.0)

def test_evidence_f1_rewards_retrieving_the_gold_paragraph():
    s = score_answers({"q1": "backpropagation"}, GOLD,
                      retrieved={"q1": ["Nets are trained with backprop."]})
    assert s["evidence_f1"] == pytest.approx(1.0)

def test_evidence_f1_punishes_retrieving_noise():
    s = score_answers({"q1": "backpropagation"}, GOLD,
                      retrieved={"q1": ["Nets are trained with backprop.", "Photons are bosons."]})
    assert s["evidence_f1"] == pytest.approx(2 / 3)   # P=1/2, R=1 -> F1=2/3

def test_evidence_f1_is_zero_when_nothing_retrieved():
    s = score_answers({"q1": "backpropagation"}, GOLD, retrieved={"q1": []})
    assert s["evidence_f1"] == 0.0
