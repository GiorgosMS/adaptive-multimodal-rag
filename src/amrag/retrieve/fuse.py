"""Reciprocal Rank Fusion (deck slide 28).

RRF combines runs without needing their scores to be commensurable -- it uses
only ranks. rrf_k=60 is the value from Cormack et al. (2009).
"""
from amrag.types import Hit


def rrf_fuse(runs: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, Hit]] = {}   # doc_id -> (best_rank, hit)

    for run in runs:
        for rank, hit in enumerate(run, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if hit.doc_id not in best or rank < best[hit.doc_id][0]:
                best[hit.doc_id] = (rank, hit)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out: list[Hit] = []
    for doc_id, score in ordered:
        src = best[doc_id][1]
        out.append(Hit(doc_id=doc_id, score=score, granularity=src.granularity,
                       modality=src.modality, page=src.page, span=src.span))
    return out
