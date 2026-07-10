"""QASPER adapter. Retrieval unit = one evidence paragraph.

Evidence in QASPER is given as verbatim paragraph strings, so we resolve them
back to paragraph indices by whitespace-normalised exact match. Unresolvable
evidence is dropped and counted -- silently ignoring it would inflate
Recall@k. Drops fall into three disjoint buckets, checked in this order:

  1. FLOAT SELECTED evidence points at a table/figure, not a body paragraph
     -- a text retriever structurally cannot retrieve these.
  2. Section-name evidence matches one of the paper's section headings, not
     an indexed paragraph -- we never index section names.
  3. Everything else ("other") is an unexplained miss and worth watching.
"""
import re
from typing import Iterator

from amrag.corpus.base import Corpus, Document, Query

_FLOAT_PREFIX = "FLOAT SELECTED"


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _doc_id(paper_id: str, para_idx: int) -> str:
    return f"{paper_id}::{para_idx}"


def _paragraphs_from_paper(paper: dict) -> list[str]:
    out: list[str] = []
    for section in paper["full_text"]["paragraphs"]:
        out.extend(section)
    return out


class QasperCorpus(Corpus):
    def __init__(self, papers: list[dict]) -> None:
        self._papers = papers
        self.dropped_float = 0
        self.dropped_section_name = 0
        self.dropped_other = 0
        self.dropped_evidence = 0

    @classmethod
    def from_raw(cls, papers: list[dict]) -> "QasperCorpus":
        return cls(papers)

    @classmethod
    def load(cls, split: str = "test") -> "QasperCorpus":
        from datasets import load_dataset
        # allenai/qasper on the Hub is a legacy loading-script dataset; the
        # installed `datasets` (>=3.0) dropped script execution entirely, so
        # we pin to the Hub's auto-converted parquet mirror. Same rows, same
        # schema -- verified by hand against the script output.
        ds = load_dataset("allenai/qasper", split=split, revision="refs/convert/parquet")
        return cls([dict(row) for row in ds])

    def documents(self) -> Iterator[Document]:
        for paper in self._papers:
            for i, para in enumerate(_paragraphs_from_paper(paper)):
                yield Document(doc_id=_doc_id(paper["id"], i), text=para,
                               meta={"paper_id": paper["id"]})

    def queries(self) -> Iterator[Query]:
        for paper in self._papers:
            qas = paper["qas"]
            for qid, question in zip(qas["question_id"], qas["question"]):
                yield Query(qid=qid, text=question)

    def qrels(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        # Accumulate counters in local variables to make them idempotent.
        # Assignment at the end is atomic -- mid-loop exceptions cannot
        # leave half-updated counters.
        dropped_float = 0
        dropped_section_name = 0
        dropped_other = 0
        dropped_evidence = 0
        for paper in self._papers:
            paras = _paragraphs_from_paper(paper)
            # Normalised-key index. Collisions keep the first (lowest-index)
            # paragraph rather than overwriting -- and never crash.
            index: dict[str, int] = {}
            for i, p in enumerate(paras):
                key = _normalise(p)
                if key not in index:
                    index[key] = i
            # Some papers have an untitled leading section, recorded as
            # `None` rather than "" -- it can never equal an evidence
            # string, so skip it rather than crashing on it.
            section_names = {_normalise(s) for s in paper["full_text"]["section_name"]
                             if s is not None}
            qas = paper["qas"]
            for qid, ann in zip(qas["question_id"], qas["answers"]):
                rels: dict[str, int] = {}
                for a in ann["answer"]:
                    for ev in a["evidence"]:
                        key = _normalise(ev)
                        if key in index:
                            rels[_doc_id(paper["id"], index[key])] = 1
                        elif key.startswith(_FLOAT_PREFIX):
                            dropped_float += 1
                            dropped_evidence += 1
                        elif key in section_names:
                            dropped_section_name += 1
                            dropped_evidence += 1
                        else:
                            dropped_other += 1
                            dropped_evidence += 1
                out[qid] = rels
        # Assign atomically at the end so counters describe the corpus,
        # not the call history.
        self.dropped_float = dropped_float
        self.dropped_section_name = dropped_section_name
        self.dropped_other = dropped_other
        self.dropped_evidence = dropped_evidence
        return out
