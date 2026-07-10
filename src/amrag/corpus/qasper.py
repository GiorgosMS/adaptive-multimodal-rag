"""QASPER adapter. Retrieval unit = one evidence paragraph.

Evidence in QASPER is given as verbatim paragraph strings, so we resolve them
back to paragraph indices by exact match. Unresolvable evidence is dropped and
counted -- silently ignoring it would inflate Recall@k.
"""
from typing import Iterator

from amrag.corpus.base import Corpus, Document, Query


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
        for paper in self._papers:
            paras = _paragraphs_from_paper(paper)
            index = {p: i for i, p in enumerate(paras)}
            qas = paper["qas"]
            for qid, ann in zip(qas["question_id"], qas["answers"]):
                rels: dict[str, int] = {}
                for a in ann["answer"]:
                    for ev in a["evidence"]:
                        if ev in index:
                            rels[_doc_id(paper["id"], index[ev])] = 1
                        else:
                            self.dropped_evidence += 1
                out[qid] = rels
        return out
