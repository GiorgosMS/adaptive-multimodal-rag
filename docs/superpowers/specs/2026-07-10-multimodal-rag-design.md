# Adaptive Multimodal RAG over Scientific Papers — Design

**Date:** 2026-07-10
**Status:** Approved design, pre-implementation
**Blueprint:** `Pharos_July_RAG_Webinar_Slides_v5.pdf` — PHAROS AI Factory / AI4Language, 07 July 2026.
Drosatos & Gyftopoulos, ILSP / Athena Research Center.

---

## 1. Motivation

The Pharos deck specifies a complete text RAG pipeline: ingestion, structure-aware chunking,
metadata, embeddings, hybrid retrieval, HyDE, RRF fusion, reranking, grounded generation with
citations, and a decomposed evaluation phase.

It has one structural blind spot. Slide 17 says *"Extract text from PDFs"* and *"preserve tables."*
Every downstream stage is text. **A research paper's load-bearing content is substantially
figures** — architecture diagrams, loss curves, qualitative grids, result tables. A text-only
pipeline discards all of it before retrieval begins.

The deck also draws, on slide 12, a Modular RAG pipeline:

```
Route → Rewrite → Search → Fuse → Rerank → Generate → Verify
```

and states its own punchline: *"the pipeline becomes adaptive rather than fixed."* It then never
builds the `Route` box.

This project builds the deck's pipeline, adds a visual retrieval path it omits, and instantiates
the `Route` box it leaves empty.

## 2. Research question

> Screenshot-native visual retrieval (ColPali-family) retrieves better on figure-grounded queries.
> Text retrieval cites more precisely and supports metadata filters.
> **Can a per-query router capture both — and does adaptivity earn its complexity?**

### Hypotheses

- **H1.** Visual retrieval beats text retrieval on figure/table-grounded queries, and loses or ties
  on prose-grounded queries.
- **H2.** Visual retrieval degrades citation granularity from passage-level to page-level.
- **H3.** A per-query modality router beats always-run-both (R0) on a quality-vs-cost frontier.

**H3 may be false.** Retrieval is cheap; running both retrievers and fusing may dominate routing.
A refuted H3 is a publishable result — *"adaptivity is overrated at this scale"* — and the design
gates on discovering it early (§7).

### The tension worth measuring

The deck spends slide 19 on *"Metadata makes retrieval controllable"* and slide 34 on
*"Cite exact passages, sections, articles — not just the document."*

**Pure screenshot RAG violates both.** You cannot filter pixels by section or publication year, and
you cannot cite a passage inside a page image. Visual retrieval buys recall and sells traceability.
This spec makes that trade a measured quantity rather than a rhetorical one.

## 3. Non-goals

Explicitly cut, to keep scope to a single implementation plan:

- **BRIGHT benchmark.** Considered and rejected: text-only, no figures, orthogonal to the modality-
  routing thesis.
- **ViDoRe v1.** Saturated (SOTA > 90 nDCG@5). Reporting it is uninformative.
- **Any user interface before M4.** The deliverable is an ablation table and a Pareto plot.
- **A vector database before M3.** M1/M2 corpora are small enough that exact MaxSim in a torch
  tensor is both correct and faster than an ANN index.
- **Custom embedding-model training.** Off-the-shelf checkpoints only.
- **Deployment** (deck slide 45). Out of scope.

## 4. Architecture

### 4.1 Module layout

```
corpus/            dataset adapters — one interface per benchmark
  base.py            iter_documents() / iter_queries() / qrels()
  qasper.py  litsearch.py  vidore_v2.py  spiqa.py  m3docvqa.py

ingest/
  parse.py         Docling → Document{sections, chunks, figures, tables, metadata}
  render.py        PDF page → PIL.Image (for the visual path)

index/
  text.py          BM25 (rank_bm25) + dense (BGE-M3)
  visual.py        ColQwen2.5 multi-vector embeddings + MaxSim
  store.py         backend interface: InMemoryStore | QdrantStore

retrieve/
  dense.py sparse.py visual.py    each: retrieve(query, k) -> list[Hit]
  hyde.py          query transformation (deck slide 27)
  fuse.py          Reciprocal Rank Fusion (deck slide 28)
  rerank.py        cross-encoder, BGE-reranker-v2-m3
  router.py        query -> modality plan          ← the contribution

generate/
  prompt.py        grounded prompt: citation format + insufficiency clause
  vlm.py           Qwen3-VL (API) — sees cited figures

eval/
  retrieval.py     Recall@k, nDCG@k, MRR, Precision@k
  answer.py        Answer-F1, Evidence-F1, L3Score
  faithfulness.py  RAGChecker claim-level diagnostics
  rejection.py     negative-rejection rate (RGB-style)
  ablate.py        config grid → results table
```

### 4.2 The key abstraction

Every retriever — sparse, dense, visual — returns the same type:

```python
@dataclass(frozen=True)
class Hit:
    doc_id: str
    page: int
    locator: Span | BBox          # character span, or figure/page bounding box
    score: float
    modality: Literal["text", "visual"]
    granularity: Literal["passage", "figure", "page"]
```

`granularity` is a **measured field, not an architectural accident**. It is what makes H2 testable:
citation quality can be computed uniformly across retrievers that natively cite at different
resolutions.

Retrievers are independently testable behind `Retriever.retrieve(query, k) -> list[Hit]`. The
router consumes retrievers; it does not know how they work.

### 4.3 Data flow

```
                    ┌─ parse (Docling) ─→ chunks + figures + tables ─→ text index
PDF ────────────────┤
                    └─ render ──────────→ page images ──────────────→ visual index

query ─→ [rewrite/HyDE] ─→ router ─→ {text | visual | both}
                                        │
                                        ├─→ sparse + dense ─┐
                                        └─→ visual ─────────┤
                                                            ├─→ RRF fuse ─→ rerank ─→ top-k Hits
                                                            │
                                                     Qwen3-VL generate ─→ answer + citations
                                                                              │
                                                                        verify / abstain
```

## 5. Benchmarks

Selected for: ground-truth evidence labels (so retrieval metrics are computable), scientific-paper
domain, and feasibility on one RTX 4070.

| Benchmark | HF / source | Size | Evidence labels? | Metrics | Used in |
|---|---|---|---|---|---|
| **QASPER** | `allenai/qasper` | 1,585 NLP papers, 5,049 Q | Yes — evidence paragraphs | Answer-F1, **Evidence-F1** | M1 |
| **LitSearch** | `princeton-nlp/LitSearch` | 64,183 papers, 597 queries | Yes — paper-level qrels | Recall@k, nDCG@k | M1 |
| **ViDoRe v2** | `vidore/vidore-benchmark-v2-*` | 4 sets, ~250 queries | Yes — page qrels | nDCG@5 | M2 |
| **SPIQA** | `google/spiqa` | Test-B 228 Q, Test-C 493 Q | Yes — within-paper figure/table refs | **L3Score**, figure top-1 acc | M2 |
| **M3DocVQA** | m3docrag.github.io (arXiv 2411.04952) | 2,441 Q, 3k PDFs, 40k pages | Yes — page-level | page-Recall@k, answer acc | M3 |

**Stretch (not committed):** M3SciQA (`yale-nlp/M3SciQA`, 1,452 Q over 70 anchor-paper clusters) —
multi-hop cross-paper figure retrieval. Deferred; it is the natural bridge to ResearchGraph.

### 5.1 Metric definitions

- **Evidence-F1 / Answer-F1** — QASPER official, token-level. Evidence is scored *before* answer,
  per deck slide 39 (*"Evaluate retrieval before evaluating answers"*).
- **nDCG@5** — ViDoRe v2 official. Sanity target: reproduce ColQwen2.5 ≈ 0.59–0.60.
- **L3Score** — SPIQA official. `softmax(logprob("Yes"), logprob("No"))` read from the judge's
  top-5 tokens; if both absent, score 0. **Requires a judge exposing token logprobs** — GPT-4o is
  convenience, not a requirement.
- **Negative-rejection rate** — fraction of queries with no supporting evidence for which the
  system abstains rather than fabricating (deck slide 35).

### 5.2 Judge protocol

Fixed judge: **DeepSeek-V4-flash** (exposes `logprobs`/`top_logprobs` on chat completions; *not* on
`deepseek-reasoner`).

**Constraint, from SPIQA Appendix B:** absolute L3Score shifts with the judge; only *relative
rankings* are stable across judges. Therefore:

1. One judge, fixed across every run in this project.
2. Report **deltas between our configurations**, never absolute L3Score compared against published
   GPT-4o numbers.
3. Cross-check ranking stability once with a local Qwen judge via vLLM (logprobs are free locally).

## 6. The router

Three escalating variants, each falsifiable against the one below it.

- **R0 — always-both + RRF.** Not a router. The baseline R1/R2 must beat.
- **R1 — zero-shot LLM classifier.** Prompt: *"does answering this query require reading a figure
  or table?"* → `{text, visual, both}`.
- **R2 — learned classifier.** Logistic regression / MiniLM features over the query, trained on
  oracle labels harvested from M1 and M2 runs.

- **Oracle router (not shippable; an upper bound).** Per query, retrospectively pick the modality
  that *did* retrieve the gold evidence. No router can beat it.

## 7. The gate

**Before any M3 code is written**, compute the oracle router on the mixed M1+M2 query set and
measure `oracle − R0`.

**Decision threshold, fixed in advance to avoid post-hoc rationalisation.** Primary metric is
nDCG@5 on the mixed query set.

- `oracle − R0 ≥ 3.0 nDCG points` → H3 is plausible. Build M3.
- `1.0 ≤ oracle − R0 < 3.0` → marginal. Build R1 only (zero-shot, ~2 days). Skip R2.
- **`oracle − R0 < 1.0` → H3 is dead.** Routing cannot beat fusion, because even a *perfect* router
  wouldn't. Stop, write it up as a negative result, skip to M4.

The threshold is committed here, before the number is known.

This costs roughly one day and uses only data M1/M2 already produced. It is the single highest-value
experiment in the project and it runs *before* the two most expensive weeks.

## 8. Milestones

| | Build | Exit criterion | Est. |
|---|---|---|---|
| **M1** | Text RAG: naïve → +BM25/RRF → +rerank → +HyDE | Ablation table on QASPER (Evidence-F1, Answer-F1) + LitSearch (Recall@k, nDCG@10); each stage's contribution isolated | 1–2 wk |
| **M2** | ColQwen2.5 visual index; Qwen3-VL generation | Per-set nDCG@5 on each *currently distributed* ViDoRe v2 subset (Insurance was withdrawn for copyright — do not expect it), and the mean over those sets, within ±0.03 of the published ColQwen2.5 band (≈0.59–0.60); SPIQA Test-B/C L3Score reported. **Also: measure actual 4070 indexing pages/s here** and replace the §12 estimate. | 2–3 wk |
| **GATE** | Oracle-router gap analysis (§7) | Explicit go/no-go on M3, written down | 1 day |
| **M3** | R0 / R1 / R2 routers | Head-to-head vs M1 and M2 on M3DocVQA page-Recall@k + answer accuracy | 2 wk |
| **M4** | RAGChecker diagnostics; abstention | Metric→fix diagnosis table (deck slide 43); negative-rejection rate reported | 1 wk |

M1 is independently shippable. It is also the **control arm** the entire text-vs-visual comparison
depends on; it is not preamble.

## 9. Stack

| Role | Choice | License | Why |
|---|---|---|---|
| PDF parse | **Docling** v2.111.0 | MIT | Figure↔caption association + TableFormer; permissive; CPU-capable |
| Figure crops (cross-check) | pdffigures2 | Apache-2.0 | 94% precision @ 90% recall on figure↔caption bboxes |
| Text dense | BGE-M3 | MIT | Multilingual, strong retrieval |
| Text sparse | BM25 (`rank_bm25`) | Apache-2.0 | Deck slide 25 lexical arm |
| Rerank | BGE-reranker-v2-m3 | Apache-2.0 | Deck slide 28 |
| Visual retrieval | **`vidore/colqwen2.5-v0.2`** | adapters MIT; Qwen backbone under Qwen Research License | ~196 vec/page (vs ColPali's 1030); ~8–9 GB fp16 fits 12 GB |
| Generation | Qwen3-VL-8B / 30B-A3B (API) | Apache-2.0 weights | Best cheap chart/figure QA; native interleaved multi-image |
| Judge | DeepSeek-V4-flash | — | Exposes logprobs; cheapest strong text reasoner |
| Store (M3 only) | Qdrant | Apache-2.0 | First-class native multivector + MaxSim |

### 9.1 Hardware constraint

ColQwen2.5 (~8–9 GB fp16) and Qwen3-VL-8B **cannot co-reside on 12 GB VRAM.**

Resolution: **index locally, generate via API.** Indexing is a one-time offline cost; generation is
per-query and cheap on Qwen3-VL.

### 9.2 Index cost

> ⚠ **DISPUTED — measure before relying on this.** Two independent web sweeps returned different
> vectors-per-page for ColQwen2.5: **~196** and **up to 768**. ColQwen uses *dynamic* image
> resolution, so both may be true for different page sizes. The figures below assume 196; at 768
> they are **4× larger** (still tractable: ~4 GB int8 for 40k pages). **M2 Task 1 is to embed ten
> real pages and print the tensor shape.** Do not plan storage on either number until then.

ColQwen2.5 emits ~196 vectors/page × 128-dim. At int8: **~25 MB per 1,000 pages.** M3DocVQA's
40k pages ≈ 1 GB int8 — trivially manageable.

Binary quantization (Vespa-documented) gives a further **~32× reduction over fp32** while retaining
~98% of nDCG@5 *when paired with a float rerank stage*. Available if needed; **not needed at this
scale.** (Per-page byte figures for binary quant are backbone-dependent — ColPali's 1030 vec/page
and ColQwen2.5's ~196 differ by 5×. Do not quote a single number across models.)

### 9.3 Provider note

**DeepSeek serves no vision model.** Verified on DeepSeek's pricing page: `deepseek-v4-flash` and
`deepseek-v4-pro` are text-only. `DeepSeek-VL2` is weights-only, not API-served. The standing
preference for DeepSeek therefore holds for the **judge**, and vision generation must go elsewhere
(Qwen3-VL, also cheap and open-weights).

## 10. Deliverable

A repository, and **one figure**:

> x-axis: index cost (bytes/page, log scale) · y-axis: retrieval quality
> points: `{text, visual, hybrid, R1, R2, oracle}`

A second panel plots citation granularity against the same x-axis, making H2 visible.

That plot argues the thesis — **including if it argues against it.**

## 11. Risks

| Risk | Mitigation |
|---|---|
| Oracle gap is small → H3 dies, M3 wasted | §7 gate. Discovered in one day, before the two most expensive weeks. |
| Judge calibration drift | One fixed judge; report deltas only; never compare to published absolutes. |
| Docling figure/caption extraction errors silently corrupt the text arm | Spot-check a stratified sample against pdffigures2 before M1 exit. |
| 12 GB VRAM contention | Offline index / API generate split (§9.1). |
| M3DocVQA indexing time | Subset first; full 40k pages only after the pipeline is stable. |
| API spend overruns | Estimated tens of dollars (§12, unverified). Meter per-run; cap SPIQA sweeps to Test-B before Test-C. |

## 12. Provenance of claims

Facts in this spec were web-verified on 2026-07-10 against arXiv, HuggingFace, GitHub, and official
pricing pages — not recalled from model memory.

**Verified:** ColPali (arXiv 2407.01449, ICLR 2025) and vector counts; ColQwen2.5 checkpoint id and
VRAM class; ViDoRe v2 (arXiv 2505.17166) metric and ColQwen2.5 score band; ViDoRe v1 saturation;
ViDoRe v3 SOTA < 65 nDCG@10; QASPER dual Answer-F1/Evidence-F1 with annotated evidence; LitSearch
sizes and official metrics; SPIQA splits, L3Score mechanics, and the Appendix-B judge-sensitivity
finding; M3DocVQA sizes; RAGChecker (NeurIPS 2024 D&B); RGB (arXiv 2309.01431, AAAI 2024); Docling
version and MIT license; DeepSeek's text-only API and `logprobs` support; Qwen3-VL licensing;
Qdrant native MaxSim.

**Explicitly unverified — do not treat as fact:**

- **RTX 4070 indexing throughput (~3–5 pages/s).** Extrapolated from A100 memory-bandwidth ratios.
  No published 4070 benchmark was found. **Measure it in M2 before relying on it.**
- **API spend estimate ("tens of dollars").** Not modeled; a guess. Meter actual spend from M1.
- **ColQwen2.5 vectors-per-page (196 vs 768).** Sources disagree; the model uses dynamic resolution.
  Every index-size figure in §9.2 rests on this. **Measured in M2 Task 1.**
- **M3DocVQA raw-PDF corpus size.** Genuinely unpublished — Bloomberg ships a Wikipedia downloader
  script, not a sized artifact. The only data point is an unofficial partial image mirror (~6.2 GB).
- **QASPER test-split size (~1,451 Q).** Derived by arithmetic (5,049 − 2,593 − 1,005), not stated
  on the dataset card. M1 Task 4 Step 5 prints the true number; trust that.
- **No published human-correlation figures exist for a cheap judge (DeepSeek/Qwen) on L3Score.**
  The only quantitative substitution evidence is SPIQA's "rankings stay consistent" claim. This is
  precisely why §5.2 forbids absolute-score comparisons.
- Several cited sources carry 2026-dated arXiv identifiers past the assistant's training cutoff and
  rest on web fetches performed during design, not on prior knowledge.

## 13. Open decisions deferred by design

- **Vector store.** InMemory through M2; Qdrant at M3. Revisit only if M3DocVQA indexing proves the
  bottleneck.
- **R2 feature set.** Chosen after M1/M2 produce oracle labels — inventing features before seeing
  the label distribution would be guessing.
- **M3SciQA.** Stretch. Evaluate only if M3 clears the gate with time remaining.
