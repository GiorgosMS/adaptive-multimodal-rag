"""HyDE (deck slide 27): generate a hypothetical answer and embed *that*,
on the theory that an answer looks more like the target passage than the
question does.

We concatenate rather than replace: a hallucinated hypothetical can drift off
the user's intent, and the original query anchors it.
"""
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
