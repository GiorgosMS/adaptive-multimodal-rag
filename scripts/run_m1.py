# scripts/run_m1.py
"""Run the ablation ladder on QASPER and LitSearch. Writes results/*.md.

The +hyde rung costs one LLM call per query. It is SKIPPED unless --with-hyde is
passed, and the skip is printed -- a silently-omitted rung that prints numbers
identical to the rung below it would read as "HyDE had no effect", which is a
fabricated finding.
"""
import argparse, os, pathlib

from amrag.corpus.litsearch import LitSearchCorpus
from amrag.corpus.qasper import QasperCorpus
from amrag.eval.ablate import ABLATION_LADDER, to_markdown
from amrag.eval.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from amrag.generate.llm import DeepSeekLLM
from amrag.index.cache import CachedEncoder
from amrag.index.text import BGEM3Encoder, BM25Retriever, DenseRetriever, Encoder
from amrag.retrieve.fuse import rrf_fuse
from amrag.retrieve.hyde import hyde_transform
from amrag.retrieve.rerank import BGEReranker, rerank

# model_id under which BGE-M3 vectors are cached (Task 17). Not the class
# name -- the HF repo id -- so a future second dense encoder can never
# collide with this one's cache directory.
_BGE_M3_MODEL_ID = "BAAI/bge-m3"


def run(corpus, encoder: Encoder, k: int = 10, limit: int = 0, with_hyde: bool = False,
        device: str = "cuda") -> str:
    docs = list(corpus.documents())
    texts = {d.doc_id: d.text for d in docs}
    qrels = corpus.qrels()
    queries = list(corpus.queries())
    if limit:
        queries = queries[:limit]

    print(f"[{len(docs)} docs, {len(queries)} queries, device={device}] encoding corpus...", flush=True)
    dense = DenseRetriever.build(docs, encoder)
    if hasattr(encoder, "hits") and hasattr(encoder, "misses"):
        print(f"cache: {encoder.hits} hit, {encoder.misses} miss", flush=True)
    sparse = BM25Retriever.build(docs)
    llm = DeepSeekLLM() if with_hyde else None

    # BGEReranker is loaded lazily, only once we reach a rung that needs it.
    # This is a memory-footprint decision, not a behavioural one: the encoder
    # alone must hold the corpus-encoding peak (the single largest GPU
    # allocation in this script), and every rung on this ladder needs dense
    # retrieval, so the encoder is built eagerly above. Only "+rerank" needs
    # the cross-encoder; constructing it up front would double resident GPU
    # memory for the entire naive/+hybrid pass for no benefit.
    reranker = None

    ladder = [c for c in ABLATION_LADDER if with_hyde or not c.hyde]
    for c in ABLATION_LADDER:
        if c not in ladder:
            print(f"SKIPPING rung {c.name!r}: needs --with-hyde (costs API calls)", flush=True)

    rows = []
    for cfg in ladder:
        if cfg.rerank and reranker is None:
            reranker = BGEReranker(device=device)
        per_query = []
        for q in queries:
            search_text = hyde_transform(q.text, llm) if cfg.hyde else q.text
            runs = []
            if cfg.dense:
                runs.append(dense.retrieve(search_text, k=100))
            if cfg.sparse:
                runs.append(sparse.retrieve(search_text, k=100))
            hits = rrf_fuse(runs, k=100) if len(runs) > 1 else runs[0]
            if cfg.rerank:
                # rerank against the ORIGINAL query, not the HyDE expansion:
                # the cross-encoder must judge relevance to what the user asked.
                hits = rerank(q.text, hits[:50], texts, reranker, k=k)
            ids = [h.doc_id for h in hits[:k]]
            g = qrels.get(q.qid, {})
            per_query.append((
                recall_at_k(ids, g, k), precision_at_k(ids, g, k),
                ndcg_at_k(ids, g, k), mrr(ids, g),
            ))
        n = len(per_query)
        rows.append({
            "config": cfg.name,
            f"recall@{k}": sum(x[0] for x in per_query) / n,
            f"precision@{k}": sum(x[1] for x in per_query) / n,
            f"ndcg@{k}": sum(x[2] for x in per_query) / n,
            "mrr": sum(x[3] for x in per_query) / n,
        })
        print(rows[-1], flush=True)
    return to_markdown(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["qasper", "litsearch"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="subset queries for a smoke run")
    ap.add_argument("--with-hyde", action="store_true", help="enable the +hyde rung (costs API calls)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu avoids contending for a GPU that other jobs are using")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the persistent embedding cache and always re-encode")
    a = ap.parse_args()

    if a.no_cache:
        encoder: Encoder = BGEM3Encoder(device=a.device)
    else:
        # Fail loudly and BEFORE loading the model: a silent fallback to
        # ~/.cache would write GBs of embeddings onto the nearly-full /
        # partition. `source scripts/env.sh` sets this; --no-cache opts out.
        amrag_data = os.environ.get("AMRAG_DATA")
        if not amrag_data:
            raise SystemExit(
                "AMRAG_DATA is not set. Run `source scripts/env.sh` first, "
                "or pass --no-cache to deliberately skip the persistent "
                "embedding cache."
            )
        cache_root = pathlib.Path(amrag_data) / "embeddings"
        encoder = CachedEncoder(BGEM3Encoder(device=a.device), _BGE_M3_MODEL_ID, cache_root)

    corpus = QasperCorpus.load("test") if a.dataset == "qasper" else LitSearchCorpus.load()
    md = run(corpus, encoder, limit=a.limit, with_hyde=a.with_hyde, device=a.device)
    suffix = "_hyde" if a.with_hyde else ""
    out = pathlib.Path(f"results/m1_{a.dataset}{suffix}.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(f"\nwrote {out}\n{md}")
