import dataclasses
import pytest
from amrag.types import Hit, Span

def test_hit_is_frozen():
    h = Hit(doc_id="d1", score=1.0, granularity="passage", modality="text")
    with pytest.raises(dataclasses.FrozenInstanceError):   # not bare Exception:
        h.score = 2.0                                      # that would pass on a typo

def test_hit_rejects_unknown_granularity():
    with pytest.raises(ValueError):
        Hit(doc_id="d1", score=1.0, granularity="paragraph", modality="text")

def test_span_rejects_inverted_range():
    with pytest.raises(ValueError):
        Span(start=5, end=3)
