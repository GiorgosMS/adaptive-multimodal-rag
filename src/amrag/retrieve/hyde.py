"""HyDE (deck slide 27): generate a hypothetical answer and embed *that*,
on the theory that an answer looks more like the target passage than the
question does.

We concatenate rather than replace: a hallucinated hypothetical can drift off
the user's intent, and the original query anchors it.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


HYDE_PROMPT = (
    "Write a short passage from a scientific paper that would directly answer "
    "the following question. Do not preface it. Two or three sentences.\n\n"
    "Question: {query}\n\nPassage:"
)


def hyde_transform(query: str, llm: LLM) -> str:
    hypothetical = llm.complete(HYDE_PROMPT.format(query=query)).strip()
    if not hypothetical:
        return query
    return f"{query}\n\n{hypothetical}"


def hyde_transform_batch(queries: list[str], llm: LLM,
                         max_workers: int = 8) -> list[str]:
    """Expand many queries concurrently, one LLM call each.

    Each HyDE call is independent and I/O-bound (a network round-trip to the
    LLM), so the whole ablation's HyDE rung is bottlenecked purely on
    sequential latency -- ~2.5s/query * 1451 queries is ~60 minutes of pure
    waiting. A bounded thread pool collapses that to (latency * n /
    max_workers) while keeping every call's result attributed to its own
    query.

    `executor.map` is used precisely because it preserves input order: the
    i-th result is the expansion of `queries[i]`, regardless of which call
    finished first. That ordering is load-bearing -- run_m1.py pairs these
    expansions back to queries positionally, so a reordering would silently
    misattribute every relevance judgement. max_workers bounds the number of
    concurrent connections; the LLM's own rate limits, not the CPU, are the
    reason to keep it modest.
    """
    if not queries:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda q: hyde_transform(q, llm), queries))
