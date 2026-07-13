# M1 pipeline improvements — design (2026-07-13)

## Context

The M1 ladder (naive → +hybrid → +rerank) is measured and published:
QASPER R@10 0.238, LitSearch R@10 0.672 / nDCG@10 0.544 — the latter within
~2 points of GritLM-7B on the LitSearch authors' own nDCG@10 numbers. The user
asked for the improvements I judged necessary. This doc records what gets
built now, what is deliberately deferred, and why.

## Principle: diagnose before optimizing

The +rerank rung re-orders the top-50 fused candidates. Fused Recall@50 is
therefore a hard ceiling on what any reranker can deliver. Every improvement
below is gated on a ceiling diagnostic (`scripts/diagnose_ceiling.py`,
fused R@{10,50,100} + per-arm R@100 on both datasets):

- If the fused pool's recall barely grows from k=50 to k=100, a deeper rerank
  buys nothing — first-stage recall is the binding constraint.
- If it grows a lot, `--rerank-depth 100` is near-free headroom.
- Per-arm numbers (dense@100 vs bm25@100) show which arm to strengthen.

## Changes being built

### 1. Ceiling diagnostic (new `scripts/diagnose_ceiling.py`)

Thin composition of already-tested parts (mirrors `run_m1.py`'s conventions:
AMRAG_DATA guard, cached encoder, all-query averaging). Writes
`results/diagnostic_{dataset}.md`.

### 2. QASPER contextual enrichment (`QasperCorpus(enrich=True)`)

The diagnosed failure mode on QASPER: questions like "what was the baseline?"
are under-specified against a bare paragraph from an unknown paper. Fix:
prepend document identity to each chunk at index time —

    {paper title}\n{section name}\n{paragraph}

- Opt-in flag, default off. Default-off preserves the existing embedding-cache
  keys and the published numbers' provenance; enrichment is a measured rung
  variant (`results/m1_qasper_enriched*.md`), never a silent change.
- `qrels()` still resolves evidence against RAW paragraph text (the dataset
  gives evidence verbatim); doc_ids (`{paper_id}::{flat_index}`) unchanged.
  Flatten order of the enriched texts must equal `_paragraphs_from_paper`'s.
- The reranker's candidate texts come from `documents()`, so the cross-encoder
  sees the enriched text too. Intentional: the reranker needs document
  identity for the same reason the embedder does.
- LitSearch already indexes title+abstract; `--enrich` with litsearch is a
  usage error (fail loudly).
- Cost: cache miss on all 20,221 QASPER paragraphs → one-time CPU re-encode.

### 3. Rerank depth as a parameter (`--rerank-depth`, default 50)

`hits[:50]` becomes `hits[:depth]`. Whether to run at 100 is decided by the
diagnostic, not assumed. Non-default depth lands in the results filename
(`_rd{depth}`) so no table is ambiguous about its configuration.

## Run plan (GPU is occupied by another job → everything `--device cpu`)

1. Diagnostics, both datasets (warm cache: index math only).
2. QASPER `--enrich`, depth 50 — isolates enrichment vs the published ladder.
3. QASPER `--enrich --rerank-depth 100` — isolates depth, if diagnostic warrants.
4. LitSearch `--rerank-depth 100` — if diagnostic warrants.

Sequential background runs; results compared rung-by-rung against the
committed `results/m1_*.md`. Figure/post updated only if numbers improve, with
config differences stated.

## Deliberately deferred (each is its own measured change later)

- **BGE-M3 self-hybrid** (its sparse lexical weights + ColBERT multi-vectors,
  three-way fusion): strongest zero-new-params upgrade, but needs a new
  encoder interface and cache schema — own task.
- **Hierarchical retrieval on QASPER** (paper-level route, then paragraphs):
  likely the biggest structural win; restructures the ladder — own task.
- **Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B swap**: multi-GB downloads and
  full re-encode; do after the cheap wins are banked.
- **LLM listwise rerank** (DeepSeek client exists): costs API money per query;
  mirrors the LitSearch paper's GPT-4o rung. Gate behind a flag like HyDE.
- **HyDE revisit**: not deferred — rejected. Measured, it lost (0.234 < 0.238).

## Testing

- Enrichment: TDD in `tests/corpus/test_qasper.py` — enriched text carries
  title+section; flatten order/doc_ids identical enrich on/off; qrels
  byte-identical enrich on/off; default stays raw.
- `--rerank-depth`/`--enrich` plumbing: exercised by the runs; the script
  stays a thin shell over tested parts, per repo convention.
