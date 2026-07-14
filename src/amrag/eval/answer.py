"""Thin shape-adapter onto the vendored official QASPER evaluator.

We do NOT score anything ourselves. `vendor/qasper_eval.py` is byte-identical to
allenai/qasper-led-baseline, and its `get_answers_and_evidence()` + `evaluate()`
define the metrics the paper reports. We only translate HuggingFace's columnar
`qas` layout into the row-wise layout upstream expects.

Every scoring subtlety -- max-over-annotators, the "Unanswerable" gold string,
the extractive/abstractive/boolean priority, the empty-gold-empty-prediction
special case -- lives upstream and must stay there.
"""
from vendor.qasper_eval import evaluate, get_answers_and_evidence


def _rowwise(paper: dict) -> dict:
    """HuggingFace gives `qas` as a dict-of-lists; upstream wants a list-of-dicts.

    One more wrinkle than the outer qas/questions flip: `qas["answers"][i]` is
    itself columnar -- `{"answer": [annotator_dict, ...], "annotation_id": [...],
    "worker_id": [...]}` -- where each element of the "answer" list is the raw
    per-annotator answer info (unanswerable/extractive_spans/free_form_answer/
    yes_no/evidence), *not* wrapped in its own "answer" key. Upstream iterates
    `qa_info["answers"]` expecting each element to be a wrapper with an
    "answer" key (`annotation_info["answer"]`), so we re-wrap each raw
    per-annotator dict here. Confirmed against a real cached QASPER paper --
    a fixture alone would not have surfaced this second level of columnarity.
    """
    qas = paper["qas"]
    return {
        "qas": [
            {"question_id": qid, "question": q,
             "answers": [{"answer": annotator} for annotator in ann["answer"]]}
            for qid, q, ann in zip(qas["question_id"], qas["question"], qas["answers"])
        ]
    }


def build_gold(papers: list[dict], text_evidence_only: bool = True) -> dict:
    """qid -> [{answer, evidence, type}, ...], one entry per annotator.

    `text_evidence_only=True` drops "FLOAT SELECTED" (figure/table) evidence,
    which a text-only retriever cannot reach. That is upstream's own flag, and
    it is what makes the Recall@k ceiling honest rather than punitive.
    """
    return get_answers_and_evidence({p["id"]: _rowwise(p) for p in papers}, text_evidence_only)


def score_answers(predictions: dict[str, dict], gold: dict) -> dict:
    """`predictions` maps qid -> {"answer": str, "evidence": [str, ...]}.

    Returns upstream's dict: "Answer F1", "Answer F1 by type", "Evidence F1",
    "Missing predictions". A qid in `gold` but absent from `predictions` scores
    0 on both metrics -- upstream counts it, it is not an exemption.
    """
    return evaluate(gold, predictions)
