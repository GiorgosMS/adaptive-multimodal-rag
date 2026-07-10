"""Adapter onto the vendored official QASPER evaluator.

BOTH metrics come from vendor/qasper_eval.py, downloaded unmodified from
allenai/qasper-led-baseline:

  token_f1_score(pred, ref)          -> Answer-F1   (SQuAD-normalised token F1)
  paragraph_f1_score(pred, gold)     -> Evidence-F1 (set F1 over paragraph strings)

We only reshape our data into their expected form. Reimplementing either would
silently break comparability with the paper -- and `paragraph_f1_score` carries
a special case (empty gold + empty prediction == 1.0, i.e. correct abstention on
an unanswerable question) that a naive set-F1 gets wrong.
"""
from vendor.qasper_eval import paragraph_f1_score, token_f1_score


def score_answers(
    predictions: dict[str, str],
    gold: dict,
    retrieved: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """`gold` maps qid -> {"answers": [str, ...], "evidence": [str, ...]}.

    A qid present in `gold` but absent from `predictions` scores 0 -- an
    unanswered question is a failure, not an exemption.

    `token_f1_score` returns int 0 on no overlap; float() keeps the return type
    homogeneous so `to_markdown`'s `:.4f` formatting never sees an int.
    """
    if not gold:
        return {"answer_f1": 0.0, "evidence_f1": 0.0}

    answer_scores, evidence_scores = [], []
    for qid, g in gold.items():
        pred = predictions.get(qid, "")
        answer_scores.append(float(max(token_f1_score(pred, ref) for ref in g["answers"])))
        if retrieved is not None:
            evidence_scores.append(
                float(paragraph_f1_score(retrieved.get(qid, []), g["evidence"]))
            )

    return {
        "answer_f1": sum(answer_scores) / len(answer_scores),
        "evidence_f1": (sum(evidence_scores) / len(evidence_scores)) if evidence_scores else 0.0,
    }
