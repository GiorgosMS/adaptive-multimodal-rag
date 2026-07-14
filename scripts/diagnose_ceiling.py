# scripts/diagnose_ceiling.py
"""Measure the fused pool's recall ceiling before touching the reranker.

The +rerank rung re-orders the top-N fused candidates, so fused Recall@N is a
hard ceiling on what ANY reranker can deliver from that pool. This script
reports where the ceiling sits (fused recall at k = 10/50/100) and how each
arm contributes (dense-only and BM25-only recall@100), so the next pipeline
change targets the stage that is actually binding instead of the one that is
fashionable.

Reuses the persistent embedding cache; with a warm cache this is index math,
no encoding. Averages over ALL queries, including those whose qrels are empty,
to stay comparable with run_m1.py's tables.
"""
import argparse
import os
import pathlib

from amrag.corpus.litsearch import LitSearchCorpus
from amrag.corpus.qasper import QasperCorpus
from amrag.eval.retrieval import recall_at_k
from amrag.index.cache import CachedEncoder
from amrag.index.text import BGEM3Encoder, BM25Retriever, DenseRetriever
from amrag.retrieve.fuse import rrf_fuse

_BGE_M3_MODEL_ID = "BAAI/bge-m3"
_KS = (10, 50, 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["qasper", "litsearch"], required=True)
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="subset queries for a smoke run")
    a = ap.parse_args()

    amrag_data = os.environ.get("AMRAG_DATA")
    if not amrag_data:
        raise SystemExit("AMRAG_DATA is not set. Run `source scripts/env.sh` first.")
    encoder = CachedEncoder(BGEM3Encoder(device=a.device), _BGE_M3_MODEL_ID,
                            pathlib.Path(amrag_data) / "embeddings")

    corpus = QasperCorpus.load("test") if a.dataset == "qasper" else LitSearchCorpus.load()
    docs = list(corpus.documents())
    qrels = corpus.qrels()
    queries = list(corpus.queries())
    if a.limit:
        queries = queries[: a.limit]

    print(f"[{len(docs)} docs, {len(queries)} queries, device={a.device}] building indexes...",
          flush=True)
    dense = DenseRetriever.build(docs, encoder)
    print(f"cache: {encoder.hits} hit, {encoder.misses} miss", flush=True)
    sparse = BM25Retriever.build(docs)

    rows = [("fused", k) for k in _KS] + [("dense", 100), ("bm25", 100)]
    sums = {r: 0.0 for r in rows}
    for i, q in enumerate(queries):
        g = qrels.get(q.qid, {})
        d = dense.retrieve(q.text, k=100)
        s = sparse.retrieve(q.text, k=100)
        fused = rrf_fuse([d, s], k=100)
        for k in _KS:
            sums[("fused", k)] += recall_at_k([h.doc_id for h in fused], g, k)
        sums[("dense", 100)] += recall_at_k([h.doc_id for h in d], g, 100)
        sums[("bm25", 100)] += recall_at_k([h.doc_id for h in s], g, 100)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(queries)} queries", flush=True)

    n = len(queries)
    lines = [f"# Ceiling diagnostic — {a.dataset} ({n} queries)", "",
             "| pool | recall |", "| --- | --- |"]
    lines += [f"| {name}@{k} | {sums[(name, k)] / n:.4f} |" for name, k in rows]
    md = "\n".join(lines) + "\n"
    out = pathlib.Path(f"results/diagnostic_{a.dataset}.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(f"\nwrote {out}\n{md}")


if __name__ == "__main__":
    main()
