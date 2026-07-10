"""Binary-relevance IR metrics. Verified against pytrec_eval in the test suite.

Convention: `run` is a list of doc_ids in descending rank order (rank 1 first);
`qrels` maps doc_id -> relevance (0 or 1). Docs absent from qrels are irrelevant.
"""
from math import log2


def _n_relevant(qrels: dict[str, int]) -> int:
    return sum(1 for v in qrels.values() if v > 0)


def recall_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    total = _n_relevant(qrels)
    if total == 0:
        return 0.0
    found = sum(1 for d in run[:k] if qrels.get(d, 0) > 0)
    return found / total


def precision_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    found = sum(1 for d in run[:k] if qrels.get(d, 0) > 0)
    return found / k


def ndcg_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    if _n_relevant(qrels) == 0:
        return 0.0
    dcg = sum(
        qrels.get(d, 0) / log2(i + 2) for i, d in enumerate(run[:k])
    )
    ideal = sorted((v for v in qrels.values() if v > 0), reverse=True)[:k]
    idcg = sum(rel / log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(run: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(run):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0
