from amrag.corpus.base import Document, Query
from amrag.corpus.litsearch import LitSearchCorpus

RAW_CORPUS = [{"corpusid": 101, "title": "Deep Nets", "abstract": "We study nets."}]
RAW_QUERIES = [{"query_set": "s", "query": "papers on nets?", "citations": [101], "specificity": 0}]


def test_document_concatenates_title_and_abstract():
    c = LitSearchCorpus.from_raw(RAW_CORPUS, RAW_QUERIES)
    assert list(c.documents()) == [
        Document(doc_id="101", text="Deep Nets\n\nWe study nets.", meta={"title": "Deep Nets"})
    ]


def test_queries_get_stable_positional_ids():
    c = LitSearchCorpus.from_raw(RAW_CORPUS, RAW_QUERIES)
    assert list(c.queries()) == [Query(qid="q0", text="papers on nets?")]


def test_qrels_come_from_citations():
    c = LitSearchCorpus.from_raw(RAW_CORPUS, RAW_QUERIES)
    assert c.qrels() == {"q0": {"101": 1}}


def test_citation_to_missing_paper_is_dropped():
    c = LitSearchCorpus.from_raw(RAW_CORPUS, [{**RAW_QUERIES[0], "citations": [101, 999]}])
    assert c.qrels() == {"q0": {"101": 1}}
    assert c.dropped_citations == 1


def test_query_with_no_resolvable_citations_gets_an_empty_dict_not_a_missing_key():
    # Every citation for this query fails to resolve -- the qid must still be
    # a key in the qrels dict, mapped to {}, never absent.
    c = LitSearchCorpus.from_raw(RAW_CORPUS, [{**RAW_QUERIES[0], "citations": [999]}])
    rels = c.qrels()
    assert "q0" in rels
    assert rels["q0"] == {}
    assert c.dropped_citations == 1


def test_dropped_citations_counter_is_idempotent_across_calls():
    """Counters must describe the corpus, not the call history.

    A driver builds the eval index with qrels(); a measurement script later
    calls qrels() again just to read the counter. They must agree -- Task 4
    shipped a bug where a dropped-item counter doubled on the second call
    because it accumulated onto `self.x` instead of a local variable.
    """
    c = LitSearchCorpus.from_raw(RAW_CORPUS, [{**RAW_QUERIES[0], "citations": [101, 999]}])
    first = c.qrels()
    dropped_after_first = c.dropped_citations
    second = c.qrels()
    dropped_after_second = c.dropped_citations
    assert first == second
    assert dropped_after_first == dropped_after_second == 1
