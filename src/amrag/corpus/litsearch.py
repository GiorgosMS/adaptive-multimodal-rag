"""LitSearch adapter. Retrieval unit = one paper (title + abstract).

Uses the `corpus_clean` config (1.26 GB). `corpus_s2orc` (full text, 1.50 GB)
is deliberately not loaded in M1 -- see the spec's disk budget.

qrels come from each query's `corpusids` list (paper-level relevance): a
corpus id that does not resolve to a corpus document is dropped and counted
in `dropped_citations` rather than silently ignored -- silently swallowing
it would inflate Recall@k. A query whose corpusids all fail to resolve
still gets an entry in `qrels()`, mapped to `{}`, never a missing key.

`corpus_clean` rows also carry a field literally named `citations`, but it
means that paper's own outgoing bibliography, not relevance to any query.
qrels() must never read it -- see test_qrels_ignore_corpus_citations_field.
The counter stays named `dropped_citations` because these labels are
citation-derived, even though the field we read is `corpusids`.
"""
from typing import Iterator

from amrag.corpus.base import Corpus, Document, Query


class LitSearchCorpus(Corpus):
    def __init__(self, corpus: list[dict], queries: list[dict]) -> None:
        self._corpus = corpus
        self._queries = queries
        self.dropped_citations = 0

    @classmethod
    def from_raw(cls, corpus: list[dict], queries: list[dict]) -> "LitSearchCorpus":
        return cls(corpus, queries)

    @classmethod
    def load(cls) -> "LitSearchCorpus":
        from datasets import load_dataset
        corpus = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
        queries = load_dataset("princeton-nlp/LitSearch", "query", split="full")
        return cls([dict(r) for r in corpus], [dict(r) for r in queries])

    def documents(self) -> Iterator[Document]:
        for row in self._corpus:
            yield Document(
                doc_id=str(row["corpusid"]),
                text=f"{row['title']}\n\n{row['abstract']}",
                meta={"title": row["title"]},
            )

    def queries(self) -> Iterator[Query]:
        for i, row in enumerate(self._queries):
            yield Query(qid=f"q{i}", text=row["query"])

    def qrels(self) -> dict[str, dict[str, int]]:
        known = {str(r["corpusid"]) for r in self._corpus}
        out: dict[str, dict[str, int]] = {}
        # Accumulate into a local variable and assign to self once at the
        # end, so the counter is a pure function of the corpus, not of how
        # many times qrels() has been called (see QASPER's dropped_evidence
        # regression: doubling the counter on a second call).
        dropped_citations = 0
        for i, row in enumerate(self._queries):
            rels: dict[str, int] = {}
            for cid in row["corpusids"]:
                if str(cid) in known:
                    rels[str(cid)] = 1
                else:
                    dropped_citations += 1
            out[f"q{i}"] = rels
        self.dropped_citations = dropped_citations
        return out
