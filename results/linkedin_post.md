# LinkedIn post — retrieval ablation (naive → hybrid → rerank → enrich → deep rerank) + SOTA comparison

## Version A (recommended — measured, researcher tone)

I built a RAG retrieval pipeline and did the boring, important part: measured what each component actually buys — then checked the result against published state of the art.

Most "I built a RAG" posts show a demo. I wanted numbers. So I built the retrieval stack one layer at a time and evaluated each layer against benchmarks with human relevance labels — QASPER and LitSearch — before touching answer generation.

The ladder (Recall@10):

QASPER    → naive 0.177 → +hybrid 0.194 → +rerank 0.238 → +enrich 0.266 → +deep rerank 0.275  (+55%)
LitSearch → naive 0.576 → +hybrid 0.604 → +rerank 0.672  (+17%)

Each rung adds exactly one thing:
• naive — dense retrieval (BGE-M3)
• +hybrid — add BM25 lexical search, fuse with Reciprocal Rank Fusion
• +rerank — re-score the top candidates with a cross-encoder
• +enrich — index every chunk as title + section + paragraph, not the bare paragraph
• +deep rerank — widen the cross-encoder's pool from 50 to 100 candidates

The last two rungs are the part I'm most proud of, because they came from a measurement, not a hunch. I first computed the retrieval ceiling: the reranker can only promote what's in its candidate pool, and on QASPER the pool held just 30% of the evidence. Diagnosis: first-stage recall was the bottleneck. The fix wasn't a bigger model — it was giving every chunk its document identity (paper title + section), so under-specified questions like "what was the baseline?" have something to match. +12% from metadata.

And one negative result, reported because honest numbers include the ones that go down: deep reranking helps QASPER (+3.6%) but *hurts* LitSearch (0.667 vs 0.672) — a deeper pool only pays when the reranker out-discriminates the added distractors. LitSearch keeps depth 50.

So how good is this, actually? The LitSearch authors (EMNLP 2024) benchmarked retrieval models on the exact same 597 queries and 64,183-paper corpus. Side by side, nDCG@10:

GTR-T5-large (0.3B)   0.29
Instructor-XL (1.2B)  0.39
E5-large-v2 (0.3B)    0.41
my pipeline (2×0.6B)  0.54  ←
GritLM-7B (7.2B)      0.56

Within ~2 points of a 7-billion-parameter embedder, ahead of every other retriever they tested by 13+ points — with nothing in the stack bigger than 0.6B parameters, on a 12 GB consumer GPU. That's the real case for retrieval pipeline engineering: stacking cheap, measured components recovers 7B-class quality from small parts.

(Fine print, because numbers without it are marketing: the paper reports nDCG@10 per query subset; I weighted their numbers by the benchmark's 155 broad / 442 specific query counts. Their rows are single retrievers; mine is a multi-stage pipeline — that's the point, but it should be said, not hidden.)

QASPER has no leaderboard for my framing — the standard task retrieves within one known paper; I made it open-corpus, searching all 20,221 paragraphs across 416 papers at once. That's why the same code scores 0.28 there and 0.67 on LitSearch: the retriever isn't the variable, the task is.

Takeaway I keep relearning: evaluate retrieval before you evaluate answers — and measure the ceiling before you optimize a stage. If the right passage never makes the shortlist, no amount of prompt engineering saves the generation step.

Blueprint was a Pharos AI RAG webinar; I rebuilt its pipeline and stress-tested it on public benchmarks. Test-driven the whole way.

Happy to share the repo / the full metrics — comment or DM.

#RAG #InformationRetrieval #MachineLearning #NLP #LLM

---

## Version B (shorter, punchier)

"I built a RAG system" is easy to say. I measured mine against published state of the art instead.

I rebuilt a RAG retrieval pipeline one component at a time and scored every version on two benchmarks with ground-truth relevance labels.

Recall@10:
• QASPER:    naive 0.177 → +hybrid 0.194 → +rerank 0.238 → +enrich 0.266 → +deep rerank 0.275  (+55%)
• LitSearch: naive 0.576 → +hybrid 0.604 → +rerank 0.672  (+17%)

The two rungs beyond reranking came from measuring, not guessing: the reranker's candidate pool held only 30% of QASPER's evidence, so the bottleneck was first-stage recall. Fix: index each chunk as title + section + paragraph instead of a bare paragraph. +12% from metadata, zero new parameters. (And deep reranking, which helps QASPER, made LitSearch slightly WORSE — measured, reported, rejected.)

The reality check: on LitSearch's published nDCG@10 — same 597 queries, same 64k-paper corpus — my pipeline scores 0.54 vs GritLM-7B at 0.56, E5-large-v2 at 0.41, Instructor-XL at 0.39, GTR at 0.29. Within ~2 points of a 7-billion-parameter embedder, with nothing in the stack bigger than 0.6B, on a consumer GPU.

Evaluate retrieval before generation, and measure the ceiling before optimizing a stage. If the answer isn't in the shortlist, the LLM can't recover it.

Repo + full metrics happy to share.

#RAG #InformationRetrieval #MachineLearning #NLP

---

## Notes

- Attach the results figure (5-rung QASPER ladder + SOTA comparison panel):
  https://claude.ai/code/artifact/2f4d7ea3-9705-4ef8-9b83-dbc1f382bdf9
  (source: results/ladder_figure.html — share from the page's menu, or screenshot the card)
- Ladder numbers are exact, from:
  results/m1_qasper.md (raw rungs 1-3), results/m1_qasper_enriched.md (rung 4),
  results/m1_qasper_enriched_rd100.md (rung 5), results/m1_litsearch.md,
  results/m1_litsearch_rd100.md (the rejected deep-rerank rung),
  results/diagnostic_{qasper,litsearch}.md (the ceiling measurements).
- Corpus sizes: QASPER 20,221 passages / 1,451 queries; LitSearch 64,183 papers / 597 queries.
- Deliberately omits HyDE (QASPER +hyde rung: R@10 0.234 < +rerank 0.238 — measured, it lost).

### Ceiling diagnostic — the numbers behind rungs 4-5

Fused pool (dense + BM25 + RRF, no rerank), Recall@k:
- QASPER:    fused@10 0.194 · fused@50 0.303 · fused@100 0.347 (dense@100 0.319, bm25@100 0.283)
- LitSearch: fused@10 0.604 · fused@50 0.780 · fused@100 0.815 (dense@100 0.789, bm25@100 0.699)

Reading: the +rerank rung was already extracting ~78% (QASPER) / ~86% (LitSearch) of its
depth-50 ceiling — the reranker wasn't the problem; the pool was. Hence +enrich (raise
first-stage recall) before +deep rerank (raise the pool depth).

### SOTA comparison — receipts (for answering comments)

Source: Ajith et al., "LitSearch: A Retrieval Benchmark for Scientific Literature Search",
EMNLP 2024, arXiv:2407.18940. Same corpus (64,183 papers, title+abstract index) and same
597 queries as our run.

Table 8 of the paper reports nDCG@10 per query subset (broad / specific):
- GTR-T5-large   23.3 / 30.4
- Instructor-XL  32.8 / 41.2
- E5-large-v2    27.1 / 45.3
- GritLM-7B      44.1 / 60.3

Table 2 gives the subset sizes: broad = 155 (120 inline + 35 author-written),
specific = 442 (231 + 211). Weighted average = (155·broad + 442·specific) / 597:
- GTR-T5-large  → 0.286
- Instructor-XL → 0.390
- E5-large-v2   → 0.406
- GritLM-7B     → 0.561
- our +rerank rung (all 597 queries): nDCG@10 = 0.544

Caveats to volunteer if asked (they make the claim stronger, not weaker):
1. The weighted average is OUR aggregation — the paper prints per-subset numbers only.
2. Paper rows are single-model retrievers with no reranking; ours is a multi-stage
   pipeline (BGE-M3 + BM25 + RRF + bge-reranker-v2-m3). The claim is pipeline-vs-model,
   i.e. cheap engineering ≈ 7B-class model, not "we beat GritLM".
3. The paper's headline recall numbers (R@20 broad / R@5 specific) use different k than
   our R@10, so nDCG@10 (their Table 8) is the only apples-to-apples metric — that's
   why the post compares on nDCG@10.
4. Their GPT-4o-reranked GritLM is stronger still (R@5 specific 79.2 / R@20 broad 75.3);
   no nDCG@10 is published for it, so it can't be placed on this chart.
5. Params: BGE-M3 ≈ 0.57B, bge-reranker-v2-m3 ≈ 0.57B, GritLM-7B ≈ 7.2B,
   Instructor-XL ≈ 1.2B, E5-large-v2 ≈ 0.34B, GTR-T5-large ≈ 0.34B.
6. QASPER enrichment (rungs 4-5) does not touch LitSearch, whose comparison numbers
   are the unchanged +rerank rung at depth 50.

QASPER: no comparable published number exists for open-corpus paragraph retrieval over
the whole test split (standard QASPER evaluates within a single given paper, via
Answer-F1/Evidence-F1). Say so plainly if asked — the honest framing is that our QASPER
numbers have no external reference point, which is exactly why the ablation ladder (each
component's marginal gain) is the claim there, not the absolute level.
