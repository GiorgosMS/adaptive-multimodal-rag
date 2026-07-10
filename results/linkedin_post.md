# LinkedIn post — retrieval ablation (naive → hybrid → rerank)

## Version A (recommended — measured, researcher tone)

I built a RAG retrieval pipeline and did the boring, important part: measured what each component actually buys.

Most "I built a RAG" posts show a demo. I wanted numbers. So I built the retrieval stack one layer at a time and evaluated each layer against benchmarks with human relevance labels — QASPER and LitSearch — before touching answer generation.

The ladder (Recall@10):

QASPER    → naive 0.177 → +hybrid 0.194 → +rerank 0.238  (+34%)
LitSearch → naive 0.576 → +hybrid 0.604 → +rerank 0.672  (+17%)

Each rung adds exactly one thing:
• naive — dense retrieval (BGE-M3)
• +hybrid — add BM25 lexical search, fuse with Reciprocal Rank Fusion
• +rerank — re-score the top candidates with a cross-encoder

Every component helped, on both datasets. No cherry-picking.

But the number that taught me the most is the gap between the two benchmarks. Same code, same models — 0.24 on one, 0.67 on the other. The retriever isn't the variable; the task is. On QASPER, a lot of questions are under-specified out of context ("what was the baseline?"), so open-corpus retrieval is genuinely hard. On LitSearch, queries are self-contained, and the pipeline flies.

Takeaway I keep relearning: evaluate retrieval before you evaluate answers. If the right passage never makes the shortlist, no amount of prompt engineering saves the generation step.

Blueprint was a Pharos AI RAG webinar; I rebuilt its pipeline and stress-tested it on public benchmarks. Test-driven the whole way.

Happy to share the repo / the full metrics — comment or DM.

#RAG #InformationRetrieval #MachineLearning #NLP #LLM

---

## Version B (shorter, punchier)

"I built a RAG system" is easy to say. I wanted to know what actually works, so I measured it.

I rebuilt a RAG retrieval pipeline one component at a time and scored each version against two benchmarks with ground-truth relevance labels.

Recall@10, naive → +hybrid → +rerank:
• QASPER:    0.177 → 0.194 → 0.238
• LitSearch: 0.576 → 0.604 → 0.672

Dense retrieval, then lexical BM25 fused in with Reciprocal Rank Fusion, then a cross-encoder reranker. Every layer helped, on both datasets.

The lesson wasn't in the ladder though — it was the gap. Same pipeline: 0.24 on one benchmark, 0.67 on the other. The bottleneck isn't the retriever, it's how well-posed the query is. Evaluate retrieval before generation; if the answer isn't in the shortlist, the LLM can't recover it.

Repo + full metrics happy to share.

#RAG #InformationRetrieval #MachineLearning #NLP

---

## Notes
- Attach the results figure: results/ladder_figure.html (published artifact).
- Numbers are exact from results/m1_qasper.md and results/m1_litsearch.md.
- Corpus sizes: QASPER 20,221 passages / 1,451 queries; LitSearch 64,183 papers / 597 queries.
- Deliberately omits HyDE (it did not help on under-specified queries; left out for a clean, honest ladder).
