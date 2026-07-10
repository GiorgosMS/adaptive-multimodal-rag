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
