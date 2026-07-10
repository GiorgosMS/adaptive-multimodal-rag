from amrag.corpus.base import Document
from amrag.index.text import BM25Retriever

DOCS = [
    Document("d1", "the cat sat on the mat", {}),
    Document("d2", "quantum entanglement of photons", {}),
    Document("d3", "a cat and a dog", {}),
]

def test_retrieves_lexically_matching_docs_first():
    r = BM25Retriever.build(DOCS)
    hits = r.retrieve("cat", k=2)
    assert {h.doc_id for h in hits} == {"d1", "d3"}

def test_hits_are_sorted_descending_by_score():
    r = BM25Retriever.build(DOCS)
    hits = r.retrieve("cat", k=3)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

def test_hits_declare_passage_granularity_and_text_modality():
    r = BM25Retriever.build(DOCS)
    h = r.retrieve("photons", k=1)[0]
    assert h.doc_id == "d2"
    assert h.granularity == "passage"
    assert h.modality == "text"

def test_k_larger_than_corpus_is_clamped():
    r = BM25Retriever.build(DOCS)
    assert len(r.retrieve("cat", k=99)) == 3

def test_satisfies_retriever_protocol():
    from amrag.types import Retriever
    assert isinstance(BM25Retriever.build(DOCS), Retriever)
