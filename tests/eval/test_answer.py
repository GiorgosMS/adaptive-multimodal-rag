"""Tests for the shape-adapter onto vendor/qasper_eval.py's official evaluate().

Every assertion here encodes the official protocol, not our own scoring logic --
these tests exist to catch any future drift back towards a hand-written loop.
"""
import pytest

from amrag.eval.answer import build_gold, score_answers


def _annotation(evidence=None, free_form_answer="", extractive_spans=None,
                 unanswerable=False, yes_no=None):
    return {
        "evidence": evidence or [],
        "free_form_answer": free_form_answer,
        "extractive_spans": extractive_spans or [],
        "unanswerable": unanswerable,
        "yes_no": yes_no,
        "highlighted_evidence": evidence or [],
    }


PAPER_TWO_ANNOTATORS = {
    "id": "p1",
    "qas": {
        "question_id": ["q1"],
        "question": ["What did they do?"],
        "answers": [
            {"answer": [
                _annotation(evidence=["A."], free_form_answer="did A"),
                _annotation(evidence=["B."], free_form_answer="did B"),
            ]}
        ],
    },
}

PAPER_UNANSWERABLE = {
    "id": "p2",
    "qas": {
        "question_id": ["q2"],
        "question": ["Is this thing on?"],
        "answers": [
            {"answer": [_annotation(unanswerable=True)]}
        ],
    },
}

PAPER_FLOAT_EVIDENCE = {
    "id": "p3",
    "qas": {
        "question_id": ["q3"],
        "question": ["What's in the table?"],
        "answers": [
            {"answer": [
                _annotation(
                    evidence=["FLOAT SELECTED: Table 1 shows results.", "Real text evidence."],
                    free_form_answer="results",
                ),
            ]}
        ],
    },
}

PAPER_ANSWER_PRIORITY = {
    "id": "p4",
    "qas": {
        "question_id": ["q4"],
        "question": ["What?"],
        "answers": [
            {"answer": [
                _annotation(free_form_answer="z", extractive_spans=["x", "y"]),
            ]}
        ],
    },
}


def test_build_gold_produces_one_reference_per_annotator():
    gold = build_gold([PAPER_TWO_ANNOTATORS])
    assert len(gold["q1"]) == 2


def test_evidence_f1_maximises_over_annotators_not_their_union():
    """The regression test that matters most.

    Annotator 1 cites ["A."], annotator 2 cites ["B."]; the prediction cites
    ["A."]. The official protocol scores each reference separately and takes
    the max: max(paragraph_f1_score(["A."], ["A."]), paragraph_f1_score(["A."], ["B."]))
    == max(1.0, 0.0) == 1.0.

    A union-scoring implementation (paragraph_f1_score(["A."], ["A.", "B."]))
    would instead give 0.667 -- a systematic under-report whenever annotators
    disagree, which is exactly the bug this task fixes.
    """
    gold = build_gold([PAPER_TWO_ANNOTATORS])
    predictions = {"q1": {"answer": "did A", "evidence": ["A."]}}
    result = score_answers(predictions, gold)
    assert result["Evidence F1"] == pytest.approx(1.0)


def test_unanswerable_question_correctly_abstained_scores_perfectly():
    gold = build_gold([PAPER_UNANSWERABLE])
    predictions = {"q2": {"answer": "Unanswerable", "evidence": []}}
    result = score_answers(predictions, gold)
    assert result["Answer F1"] == pytest.approx(1.0)
    assert result["Evidence F1"] == pytest.approx(1.0)


def test_empty_string_answer_to_unanswerable_question_scores_zero():
    """token_f1_score has no empty-empty special case (unlike paragraph_f1_score):
    an empty prediction against the literal gold string "Unanswerable" is 0, not 1.
    """
    gold = build_gold([PAPER_UNANSWERABLE])
    predictions = {"q2": {"answer": "", "evidence": []}}
    result = score_answers(predictions, gold)
    assert result["Answer F1"] == 0


def test_missing_prediction_is_counted_and_scores_zero():
    gold = build_gold([PAPER_UNANSWERABLE])
    result = score_answers({}, gold)
    assert result["Missing predictions"] == 1
    assert result["Answer F1"] == pytest.approx(0.0)
    assert result["Evidence F1"] == pytest.approx(0.0)


def test_text_evidence_only_true_drops_float_selected_evidence():
    gold = build_gold([PAPER_FLOAT_EVIDENCE], text_evidence_only=True)
    assert gold["q3"][0]["evidence"] == ["Real text evidence."]


def test_text_evidence_only_false_keeps_float_selected_evidence():
    gold = build_gold([PAPER_FLOAT_EVIDENCE], text_evidence_only=False)
    assert "FLOAT SELECTED: Table 1 shows results." in gold["q3"][0]["evidence"]


def test_answer_priority_prefers_extractive_spans_joined_with_comma_space():
    gold = build_gold([PAPER_ANSWER_PRIORITY])
    assert gold["q4"][0]["answer"] == "x, y"
