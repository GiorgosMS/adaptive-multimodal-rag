"""Grounded prompt construction (deck slides 33-35)."""
from amrag.types import Hit

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

_TEMPLATE = """\
Answer the question using ONLY the numbered evidence below.

Rules:
- Cite every claim with the bracketed number of the evidence supporting it, e.g. [1].
- Do not use knowledge that is absent from the evidence.
- If the evidence does not contain enough information to answer, reply with
  exactly {sentinel} and nothing else. Saying {sentinel} is always preferable
  to an unsupported answer.

Evidence:
{evidence}

Question: {query}

Answer:"""


def build_grounded_prompt(query: str, hits: list[Hit], doc_texts: dict[str, str]) -> str:
    if hits:
        evidence = "\n".join(
            f"[{i}] {doc_texts[h.doc_id]}" for i, h in enumerate(hits, start=1)
        )
    else:
        evidence = "(no evidence retrieved)"
    return _TEMPLATE.format(sentinel=INSUFFICIENT_EVIDENCE, evidence=evidence, query=query)
