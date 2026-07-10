from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    meta: dict


@dataclass(frozen=True)
class Query:
    qid: str
    text: str


class Corpus(Protocol):
    def documents(self) -> Iterator[Document]: ...
    def queries(self) -> Iterator[Query]: ...
    def qrels(self) -> dict[str, dict[str, int]]: ...
