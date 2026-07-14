from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Granularity = Literal["passage", "figure", "page"]
Modality = Literal["text", "visual"]

_GRANULARITIES = {"passage", "figure", "page"}
_MODALITIES = {"text", "visual"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"inverted span: {self.start} > {self.end}")


@dataclass(frozen=True)
class Hit:
    """A single retrieved unit of evidence.

    `granularity` records the finest unit this retriever can honestly cite.
    Visual retrievers cite whole pages; text retrievers cite spans. Hypothesis
    H2 of the spec is a claim about this field, so it must never be inflated.
    """
    doc_id: str
    score: float
    granularity: Granularity
    modality: Modality
    page: int | None = None
    span: Span | None = None

    def __post_init__(self) -> None:
        if self.granularity not in _GRANULARITIES:
            raise ValueError(f"bad granularity: {self.granularity!r}")
        if self.modality not in _MODALITIES:
            raise ValueError(f"bad modality: {self.modality!r}")


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return up to k hits, sorted by descending score."""
        ...
