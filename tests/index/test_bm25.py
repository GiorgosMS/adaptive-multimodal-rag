from amrag.corpus.base import Document
from amrag.index.text import BM25Retriever, _tokenize

DOCS = [
    Document("d1", "the cat sat on the mat", {}),
    Document("d2", "quantum entanglement of photons", {}),
    Document("d3", "a cat and a dog", {}),
]

# Real corpus text and real questions carry punctuation; every fixture above
# does not, which is why this went unnoticed until the sparse arm was measured
# alone on QASPER. `lower().split()` welds punctuation onto tokens, so a
# document's "BERT," never matches a query's "BERT", and a question ending in
# "...F1 score?" searches for the token "score?", which occurs nowhere.
# Measured on QASPER (20,221 paragraphs, 1,451 queries), fixing this lifts
# BM25's standalone Recall@10 from 0.0997 to 0.1446 -- +45% relative.
#
# p0 is a decoy in index position 0. It is load-bearing: when no query token
# matches anything, every BM25 score is 0.0 and the stable sort returns
# document 0. A target placed first would therefore be "retrieved" even by a
# totally broken tokenizer, and the test would pass for the wrong reason.
PUNCT_DOCS = [
    Document("p0", "An unrelated decoy document about the weather.", {}),
    Document("p1", "We fine-tune BERT, then evaluate it.", {}),
    Document("p2", "Photosynthesis converts light into sugar.", {}),
]


def test_tokenizer_strips_punctuation_from_tokens():
    assert _tokenize("F1 score?") == ["f1", "score"]
    assert _tokenize("BERT, and GPT.") == ["bert", "and", "gpt"]


def test_document_term_followed_by_punctuation_is_still_matchable():
    """The doc says "BERT," -- the query says "BERT"."""
    r = BM25Retriever.build(PUNCT_DOCS)
    hit = r.retrieve("BERT", k=1)[0]
    assert hit.doc_id == "p1"
    assert hit.score > 0.0, "a real lexical match must score above the all-zero tie"


def test_query_term_followed_by_punctuation_still_matches():
    """The query says "photosynthesis?" -- the doc says "Photosynthesis"."""
    r = BM25Retriever.build(PUNCT_DOCS)
    hit = r.retrieve("photosynthesis?", k=1)[0]
    assert hit.doc_id == "p2"
    assert hit.score > 0.0

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
