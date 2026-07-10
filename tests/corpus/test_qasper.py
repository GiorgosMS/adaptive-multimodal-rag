import pytest
from amrag.corpus.base import Document, Query
from amrag.corpus.qasper import QasperCorpus, _paragraphs_from_paper, _doc_id

RAW_PAPER = {
    "id": "p1",
    "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta.", "Gamma delta."]]},
    "qas": {
        "question": ["What is alpha?"],
        "question_id": ["q1"],
        "answers": [{"answer": [{"evidence": ["Alpha beta."], "free_form_answer": "beta",
                                 "extractive_spans": [], "unanswerable": False,
                                 "yes_no": None, "highlighted_evidence": ["Alpha beta."]}]}],
    },
}

def test_doc_id_is_paper_scoped():
    assert _doc_id("p1", 0) == "p1::0"

def test_paragraphs_are_flattened_in_order():
    assert _paragraphs_from_paper(RAW_PAPER) == ["Alpha beta.", "Gamma delta."]

def test_qrels_point_at_the_evidence_paragraph():
    c = QasperCorpus.from_raw([RAW_PAPER])
    assert c.qrels() == {"q1": {"p1::0": 1}}

def test_documents_and_queries_round_trip():
    c = QasperCorpus.from_raw([RAW_PAPER])
    docs = list(c.documents())
    assert docs[0] == Document(doc_id="p1::0", text="Alpha beta.", meta={"paper_id": "p1"})
    assert list(c.queries()) == [Query(qid="q1", text="What is alpha?")]

def test_evidence_not_matching_any_paragraph_is_dropped_not_crashed():
    paper = {**RAW_PAPER}
    paper["qas"] = {**RAW_PAPER["qas"]}
    paper["qas"]["answers"] = [{"answer": [{"evidence": ["Nowhere."], "free_form_answer": "x",
                                            "extractive_spans": [], "unanswerable": False,
                                            "yes_no": None, "highlighted_evidence": ["Nowhere."]}]}]
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {}}


def _ans(evidence: str) -> dict:
    return {"answer": [{"evidence": [evidence], "free_form_answer": "x",
                        "extractive_spans": [], "unanswerable": False,
                        "yes_no": None, "highlighted_evidence": [evidence]}]}


def test_whitespace_only_mismatch_resolves_to_the_paragraph():
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta.", "Gamma delta."]]},
        "qas": {
            "question": ["What is alpha?"],
            "question_id": ["q1"],
            "answers": [_ans("Alpha   beta.\n")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {"p1::0": 1}}


def test_float_selected_evidence_is_dropped_and_counted_only_as_float():
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta.", "Gamma delta."]]},
        "qas": {
            "question": ["What is alpha?"],
            "question_id": ["q1"],
            "answers": [_ans("FLOAT SELECTED: Table 1: results")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {}}
    assert c.dropped_float == 1
    assert c.dropped_section_name == 0
    assert c.dropped_other == 0


def test_section_name_evidence_is_dropped_and_counted_only_as_section_name():
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Experiments ::: Human Evaluation Metrics"],
                      "paragraphs": [["Alpha beta."]]},
        "qas": {
            "question": ["What is alpha?"],
            "question_id": ["q1"],
            "answers": [_ans("Experiments ::: Human Evaluation Metrics")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {}}
    assert c.dropped_section_name == 1
    assert c.dropped_float == 0
    assert c.dropped_other == 0


def test_unmatched_evidence_is_dropped_and_counted_only_as_other():
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta.", "Gamma delta."]]},
        "qas": {
            "question": ["What is alpha?"],
            "question_id": ["q1"],
            "answers": [_ans("Nowhere.")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {}}
    assert c.dropped_other == 1
    assert c.dropped_float == 0
    assert c.dropped_section_name == 0


def test_dropped_evidence_equals_sum_of_the_three_buckets():
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta."]]},
        "qas": {
            "question": ["Q?"],
            "question_id": ["q1"],
            "answers": [{"answer": [
                {"evidence": ["FLOAT SELECTED: Table 1"], "free_form_answer": "x",
                 "extractive_spans": [], "unanswerable": False, "yes_no": None,
                 "highlighted_evidence": []},
                {"evidence": ["Intro"], "free_form_answer": "x",
                 "extractive_spans": [], "unanswerable": False, "yes_no": None,
                 "highlighted_evidence": []},
                {"evidence": ["Nowhere."], "free_form_answer": "x",
                 "extractive_spans": [], "unanswerable": False, "yes_no": None,
                 "highlighted_evidence": []},
            ]}],
        },
    }
    c = QasperCorpus.from_raw([paper])
    c.qrels()
    assert c.dropped_evidence == c.dropped_float + c.dropped_section_name + c.dropped_other
    assert c.dropped_evidence == 3


def test_colliding_normalised_paragraphs_keep_first_and_do_not_crash():
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha  beta.", "Alpha beta."]]},
        "qas": {
            "question": ["Q?"],
            "question_id": ["q1"],
            "answers": [_ans("Alpha beta.")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {"p1::0": 1}}


def test_none_section_name_does_not_crash_and_evidence_still_drops_as_other():
    # Real QASPER data has untitled leading sections recorded as `None`
    # rather than "" -- must not crash when normalising section names.
    paper = {
        "id": "p1",
        "full_text": {"section_name": [None, "Intro"], "paragraphs": [["x"], ["Alpha beta."]]},
        "qas": {
            "question": ["Q?"],
            "question_id": ["q1"],
            "answers": [_ans("Nowhere.")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {}}
    assert c.dropped_other == 1


def test_raw_papers_returns_same_list_object_as_constructed_with():
    papers = [RAW_PAPER]
    c = QasperCorpus.from_raw(papers)
    assert c.raw_papers() is papers


def test_qrels_counters_are_idempotent_across_calls():
    """Counters must describe the corpus, not the call history.

    A driver builds the eval index with qrels(); a measurement script later
    calls qrels() again just to read the counters. They must agree.
    """
    paper = {
        "id": "p1",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta."]]},
        "qas": {
            "question": ["Q?"],
            "question_id": ["q1"],
            "answers": [_ans("Nowhere.")],
        },
    }
    c = QasperCorpus.from_raw([paper])
    first = c.qrels()
    counters_after_first = (c.dropped_float, c.dropped_section_name,
                            c.dropped_other, c.dropped_evidence)
    second = c.qrels()
    counters_after_second = (c.dropped_float, c.dropped_section_name,
                             c.dropped_other, c.dropped_evidence)
    assert first == second
    assert counters_after_first == counters_after_second
    # Invariant: dropped_evidence == sum of the three buckets
    assert c.dropped_evidence == c.dropped_float + c.dropped_section_name + c.dropped_other
