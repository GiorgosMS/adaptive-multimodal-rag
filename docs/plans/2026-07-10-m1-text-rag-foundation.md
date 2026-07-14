# M1: Text RAG Foundation — Implementation Plan

**Goal:** Build and evaluate the Pharos deck's text RAG pipeline (naïve → +hybrid/RRF → +rerank → +HyDE) on QASPER and LitSearch, producing an ablation table that isolates each stage's contribution.

**Architecture:** A `Retriever` protocol returning a uniform `Hit` type lets sparse, dense, fused, and reranked retrievers compose and be swapped without any consumer knowing which is which. Evaluation is decomposed — retrieval metrics are computed and gated *before* any answer is generated (deck slide 39). Official metrics are **vendored, never reimplemented**.

**Tech Stack:** Python 3.13, PyTorch, `rank_bm25`, `sentence-transformers` (BGE-M3), `FlagEmbedding` reranker, `datasets`, `pytest`, `pytrec_eval` (metric cross-check), DeepSeek API (text generation + judging).

## Global Constraints

Copied verbatim from `docs/specs/2026-07-10-multimodal-rag-design.md`:

- **Never reimplement an official metric.** Vendor the upstream evaluator. Reimplemented metrics produce numbers that are not comparable to published results.
- **One fixed judge across every run:** `deepseek-v4-flash`. Report deltas between our configurations; **never** compare absolute scores against published GPT-4o numbers.
- **Evaluate retrieval before evaluating answers.** An answer metric computed on top of unmeasured retrieval is undiagnosable.
- **No vector database in M1/M2.** Exact search in a torch tensor is correct and faster at this corpus size.
- **No UI.** The deliverable is a table and a plot.
- **`Hit.granularity` is a measured field**, not an incidental one. Every retriever must set it honestly.
- **DeepSeek serves no vision model.** Text path only in M1. Vision generation (M2) goes to Qwen3-VL.
- Python 3.13.5; system CUDA for an RTX 4070 (12 GB).

## Disk & Environment Budget

Measured 2026-07-10. **`/` (home) has only 35 GB free; the project drive has 407 GB.** Everything cacheable must live on the project drive.

**Single-folder rule (user requirement):** every downloaded byte — HF models, HF datasets, torch
weights, derived artefacts — lands under `<repo>/_cache/`. `rm -rf _cache` reclaims all of it and
leaves a working repo. `scripts/disk.sh` reports what is in there. `_cache/` is git-ignored; nothing
else in the tree grows.

Verified: the drive is `ntfs3` and **supports symlinks and hardlinks**, so the HuggingFace cache works there unmodified. Do *not* set `HF_HUB_DISABLE_SYMLINKS`.

| Item | Size | Note |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` (M2 base) | 7.51 GB | not needed in M1 |
| `vidore/colqwen2.5-v0.2` adapter (M2) | 0.26 GB | not needed in M1 |
| `BAAI/bge-m3` | ~~2.29 GB~~ → **4.3 GB MEASURED** | ⚠️ `sentence-transformers` pulls **both** `model.safetensors` (2.2 GB) *and* the legacy `pytorch_model.bin` (2.2 GB) — the same weights twice. (Not an ONNX export, as earlier research claimed.) To avoid, pre-fetch with `snapshot_download(..., ignore_patterns=["pytorch_model.bin", "onnx/*"])` and load from the local dir. Not worth doing at 390 GB free. |
| `BAAI/bge-reranker-v2-m3` | 2.29 GB | |
| `allenai/qasper` | 14.7 MB dl → 36.8 MB arrow | |
| `princeton-nlp/LitSearch` `query` | 49 KB | |
| `princeton-nlp/LitSearch` `corpus_clean` | 1.26 GB | title+abstract. **This is the M1 corpus.** |
| `princeton-nlp/LitSearch` `corpus_s2orc` | 1.50 GB | full text. **Not needed in M1.** |
| **M1 total** | ~~≈ 6.0 GB~~ → **≈ 8.3 GB** | Revised after measurement. `_cache/` measured at **11 GB** after Task 7 (includes pip wheel cache ~2.8 GB). |
| ViDoRe v2 (M2) | ≈ 2.41 GB | |
| `google/spiqa` **test splits only** | **443 MB** | ⚠ **full repo is 34.86 GB** — `train_val Images.zip` alone is 32.02 GB. Always use `allow_patterns`. |
| M3DocVQA PDFs (M3) | **unpublished** | Bloomberg ships a downloader script, not a sized artifact. Budget ~10–20 GB, verify empirically. |
| Page-image cache | **0 GB** | Do not cache. Render → embed → discard; keep PDFs; re-render top-k at query time. |
| ColQwen2.5 embeddings, 40k pages (M3) | **1–8 GB** | ⚠ depends on vectors/page, which is **disputed (196 vs 768)**. M2 Task 1 measures it. |

**Whole-project estimate: ~60 GB, of which ~6 GB is M1.** The M3DocVQA PDF corpus is the only genuinely unknown term.

---

## File Structure

```
src/amrag/
  types.py          Hit, Span, Retriever protocol            — Task 2
  corpus/
    base.py         Corpus protocol: documents/queries/qrels — Task 4
    qasper.py       QASPER adapter                           — Task 4
    litsearch.py    LitSearch adapter                        — Task 5
  index/
    text.py         BM25Retriever, DenseRetriever            — Tasks 6, 7
  retrieve/
    fuse.py         RRFFuser                                 — Task 8
    rerank.py       CrossEncoderReranker                     — Task 9
    hyde.py         HyDETransform                            — Task 10
  generate/
    llm.py          DeepSeek client (protocol + impl)        — Task 11
    prompt.py       grounded prompt w/ insufficiency clause  — Task 11
  eval/
    retrieval.py    recall_at_k, ndcg_at_k, mrr, p_at_k      — Task 3
    answer.py       vendored QASPER evaluator + adapter      — Task 12
    ablate.py       config grid → markdown table             — Task 13
vendor/
  qasper_eval.py    upstream QASPER scorer, unmodified       — Task 12
tests/               mirrors src/amrag/
```

Rationale: `index/` owns *how things are stored and searched*; `retrieve/` owns *how results are transformed*. Both speak only `Hit`. A file per responsibility, each small enough to hold in context.

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`, `src/amrag/__init__.py`, `tests/test_smoke.py`, `.env.example`, `scripts/env.sh`, `scripts/disk.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `amrag` package; `pytest` runs green.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import amrag
    assert amrag.__version__ == "0.1.0"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag'`

- [ ] **Step 3: Create the package and config**

```toml
# pyproject.toml
[project]
name = "amrag"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
  "torch>=2.5", "datasets>=3.0", "rank-bm25>=0.2.2",
  "sentence-transformers>=3.0", "FlagEmbedding>=1.3",
  "pytrec-eval-terrier>=0.5.6", "openai>=1.40",
  "numpy>=2.0", "tqdm>=4.66",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/amrag"]
[tool.pytest.ini_options]
markers = ["slow: needs real model weights or network"]
addopts = "-m 'not slow'"
```

```python
# src/amrag/__init__.py
__version__ = "0.1.0"
```

`scripts/env.sh` — **this is the file that keeps `/` from filling up:**

```bash
#!/usr/bin/env bash
# Source before any work.  /  has ~35 GB free; the project drive has ~407 GB.
#
# EVERY downloaded byte lands under a single folder: $AMRAG_CACHE.
#   rm -rf "$AMRAG_CACHE"     <- reclaims all of it; the repo still works.
# Nothing else in the tree grows. Do not let any tool default to ~/.cache.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AMRAG_CACHE="$PROJECT_ROOT/_cache"
export HF_HOME="$AMRAG_CACHE/huggingface"       # models + datasets
export TORCH_HOME="$AMRAG_CACHE/torch"
export PIP_CACHE_DIR="$AMRAG_CACHE/pip"         # torch wheels are GBs
export AMRAG_DATA="$AMRAG_CACHE/data"           # our own derived artefacts

mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AMRAG_DATA" || {
  echo "FATAL: cannot create $AMRAG_CACHE" >&2; return 1 2>/dev/null || exit 1; }

# The project path contains a space ("Personal Projects"). Fail loudly now
# rather than inside a subprocess that forgot to quote.
[ -w "$HF_HOME" ] || { echo "FATAL: $HF_HOME not writable" >&2; return 1 2>/dev/null || exit 1; }

echo "AMRAG_CACHE=$AMRAG_CACHE"
```

`scripts/disk.sh` — so the cache is never a mystery:

```bash
#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" >/dev/null
echo "Everything below is safe to delete (rm -rf \"\$AMRAG_CACHE\"):"
du -sh "$AMRAG_CACHE"/* 2>/dev/null | sort -h
echo "---"
du -sh "$AMRAG_CACHE"
df -h --output=target,avail "$AMRAG_CACHE" | tail -1
```

```bash
# .env.example
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

- [ ] **Step 4: Create venv with `--copies` and install**

NTFS handles symlinks, but venv symlinks into `/usr/bin` are fragile across the mount. Use copies:

```bash
cd "/media/giorgos-miltos-sandalis/8C645C0B645BF684/Personal Projects/adaptive-multimodal-rag"
python3 -m venv --copies .venv
source .venv/bin/activate
source scripts/env.sh
pip install -e ".[dev]"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Silence NTFS filemode noise, then commit**

```bash
git config core.filemode false
printf '%s\n' '.venv/' '_cache/' '.env' >> .gitignore    # results/ IS committed
chmod +x scripts/env.sh scripts/disk.sh
git add pyproject.toml src tests scripts .env.example .gitignore
git commit -m "feat: project scaffold; all bulk data confined to _cache/"
```

Sanity-check the confinement before moving on — nothing may land in `$HOME`:

```bash
source scripts/env.sh
python -c "import huggingface_hub.constants as c; print(c.HF_HUB_CACHE)"
# must print a path under .../adaptive-multimodal-rag/_cache/huggingface
# NB: `huggingface_hub.constants` must be imported directly -- on hf_hub >=1.23
# there is no lazy top-level `constants` attribute, so `h.constants` AttributeErrors.
```

---

### Task 2: Core types — `Hit` and the `Retriever` protocol

**Files:**
- Create: `src/amrag/types.py`, `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Span(start:int, end:int)`; `Hit(doc_id:str, score:float, granularity:Literal["passage","figure","page"], modality:Literal["text","visual"], page:int|None=None, span:Span|None=None)`; `Retriever` protocol with `retrieve(query:str, k:int) -> list[Hit]`. **Every later task depends on these exact names.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.types'`

- [ ] **Step 3: Implement**

```python
# src/amrag/types.py
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Granularity = Literal["passage", "figure", "page"]
Modality = Literal["text", "visual"]

_GRANULARITIES = {"passage", "figure", "page"}
_MODALITIES = {"text", "visual"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"inverted span: {self.start} > {self.end}")


@dataclass(frozen=True)
class Hit:
    """A single retrieved unit of evidence.

    `granularity` records the finest unit this retriever can honestly cite.
    Visual retrievers cite whole pages; text retrievers cite spans. Hypothesis
    H2 of the spec is a claim about this field, so it must never be inflated.
    """
    doc_id: str
    score: float
    granularity: Granularity
    modality: Modality
    page: int | None = None
    span: Span | None = None

    def __post_init__(self) -> None:
        if self.granularity not in _GRANULARITIES:
            raise ValueError(f"bad granularity: {self.granularity!r}")
        if self.modality not in _MODALITIES:
            raise ValueError(f"bad modality: {self.modality!r}")


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return up to k hits, sorted by descending score."""
        ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_types.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/types.py tests/test_types.py
git commit -m "feat: Hit/Span types and Retriever protocol"
```

---

### Task 3: Retrieval metrics, cross-checked against `pytrec_eval`

This is the highest-leverage test in M1. If these metrics are wrong, every number in the project is wrong and nothing downstream will reveal it.

**Files:**
- Create: `src/amrag/eval/retrieval.py`, `src/amrag/eval/__init__.py`, `tests/eval/test_retrieval.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `recall_at_k(run:list[str], qrels:dict[str,int], k:int) -> float`; `precision_at_k(...) -> float`; `ndcg_at_k(...) -> float`; `mrr(run:list[str], qrels:dict[str,int]) -> float`. `run` is doc_ids in rank order; `qrels` maps doc_id → binary relevance.

- [ ] **Step 1: Write the failing test**

Fixture values below were computed by execution, not by hand:

```python
# tests/eval/test_retrieval.py
import pytest
from amrag.eval.retrieval import recall_at_k, precision_at_k, ndcg_at_k, mrr

QRELS = {"d1": 1, "d2": 1}
RUN = ["d3", "d1", "d2"]

def test_recall_at_k():
    assert recall_at_k(RUN, QRELS, 3) == pytest.approx(1.0)
    assert recall_at_k(RUN, QRELS, 2) == pytest.approx(0.5)

def test_precision_at_k():
    assert precision_at_k(RUN, QRELS, 3) == pytest.approx(2 / 3)

def test_ndcg_at_k():
    # DCG@3  = 0/log2(2) + 1/log2(3) + 1/log2(4) = 1.130930
    # IDCG@3 = 1/log2(2) + 1/log2(3)             = 1.630930
    assert ndcg_at_k(RUN, QRELS, 3) == pytest.approx(0.693426, abs=1e-6)

def test_mrr_uses_first_relevant_rank():
    assert mrr(RUN, QRELS) == pytest.approx(0.5)

def test_no_relevant_docs_scores_zero():
    assert ndcg_at_k(["x"], QRELS, 3) == 0.0
    assert mrr(["x"], QRELS) == 0.0
    assert recall_at_k(["x"], QRELS, 3) == 0.0

def test_empty_qrels_does_not_divide_by_zero():
    assert recall_at_k(RUN, {}, 3) == 0.0
    assert ndcg_at_k(RUN, {}, 3) == 0.0

def test_ndcg_matches_pytrec_eval():
    """Our nDCG must agree with the reference IR implementation.

    Valid ONLY for binary relevance: trec_eval uses exponential gain (2^rel - 1),
    ours uses linear gain (rel). For rel in {0,1} these coincide (2^1-1 == 1).
    If graded relevance is ever introduced, this test will break -- correctly.
    """
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator({"q1": QRELS}, {"ndcg_cut_3"})
    scored = ev.evaluate({"q1": {d: float(len(RUN) - i) for i, d in enumerate(RUN)}})
    assert ndcg_at_k(RUN, QRELS, 3) == pytest.approx(scored["q1"]["ndcg_cut_3"], abs=1e-6)

def test_all_qrels_are_binary():
    """Guards the assumption the pytrec_eval cross-check depends on."""
    assert set(QRELS.values()) <= {0, 1}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/eval/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.eval'`

- [ ] **Step 3: Implement**

```python
# src/amrag/eval/retrieval.py
"""Binary-relevance IR metrics. Verified against pytrec_eval in the test suite.

Convention: `run` is a list of doc_ids in descending rank order (rank 1 first);
`qrels` maps doc_id -> relevance (0 or 1). Docs absent from qrels are irrelevant.
"""
from math import log2


def _n_relevant(qrels: dict[str, int]) -> int:
    return sum(1 for v in qrels.values() if v > 0)


def recall_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    total = _n_relevant(qrels)
    if total == 0:
        return 0.0
    found = sum(1 for d in run[:k] if qrels.get(d, 0) > 0)
    return found / total


def precision_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    found = sum(1 for d in run[:k] if qrels.get(d, 0) > 0)
    return found / k


def ndcg_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    if _n_relevant(qrels) == 0:
        return 0.0
    dcg = sum(
        qrels.get(d, 0) / log2(i + 2) for i, d in enumerate(run[:k])
    )
    ideal = sorted((v for v in qrels.values() if v > 0), reverse=True)[:k]
    idcg = sum(rel / log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(run: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(run):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0
```

Also create an empty `src/amrag/eval/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/eval/test_retrieval.py -v`
Expected: 7 passed. **If `test_ndcg_matches_pytrec_eval` fails, our implementation is wrong — fix ours, never the assertion.**

- [ ] **Step 5: Commit**

```bash
git add src/amrag/eval tests/eval
git commit -m "feat: retrieval metrics cross-checked against pytrec_eval"
```

---

### Task 4: Corpus protocol and QASPER adapter

**Files:**
- Create: `src/amrag/corpus/__init__.py`, `src/amrag/corpus/base.py`, `src/amrag/corpus/qasper.py`, `tests/corpus/test_qasper.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Document(doc_id:str, text:str, meta:dict)`; `Query(qid:str, text:str)`; `Corpus` protocol with `documents() -> Iterator[Document]`, `queries() -> Iterator[Query]`, `qrels() -> dict[str, dict[str,int]]` (qid → doc_id → rel). `QasperCorpus(split:str)`.

In QASPER a "document" is **one evidence paragraph**, keyed `f"{paper_id}::{para_idx}"`. That is the retrieval unit, so that is what qrels index.

- [ ] **Step 1: Write the failing test**

```python
# tests/corpus/test_qasper.py
import pytest
from amrag.corpus.base import Document, Query
from amrag.corpus.qasper import QasperCorpus, _paragraphs_from_paper, _doc_id

RAW_PAPER = {
    "id": "p1",
    "full_text": {"section_name": ["Intro"], "paragraphs": [["Alpha beta.", "Gamma delta."]]},
    "qas": {
        "question": ["What is alpha?"],
        "question_id": ["q1"],
        "answers": [{"answer": [{"evidence": ["Alpha beta."], "free_form_answer": "beta",
                                 "extractive_spans": [], "unanswerable": False,
                                 "yes_no": None, "highlighted_evidence": ["Alpha beta."]}]}],
    },
}

def test_doc_id_is_paper_scoped():
    assert _doc_id("p1", 0) == "p1::0"

def test_paragraphs_are_flattened_in_order():
    assert _paragraphs_from_paper(RAW_PAPER) == ["Alpha beta.", "Gamma delta."]

def test_qrels_point_at_the_evidence_paragraph():
    c = QasperCorpus.from_raw([RAW_PAPER])
    assert c.qrels() == {"q1": {"p1::0": 1}}

def test_documents_and_queries_round_trip():
    c = QasperCorpus.from_raw([RAW_PAPER])
    docs = list(c.documents())
    assert docs[0] == Document(doc_id="p1::0", text="Alpha beta.", meta={"paper_id": "p1"})
    assert list(c.queries()) == [Query(qid="q1", text="What is alpha?")]

def test_evidence_not_matching_any_paragraph_is_dropped_not_crashed():
    paper = {**RAW_PAPER}
    paper["qas"] = {**RAW_PAPER["qas"]}
    paper["qas"]["answers"] = [{"answer": [{"evidence": ["Nowhere."], "free_form_answer": "x",
                                            "extractive_spans": [], "unanswerable": False,
                                            "yes_no": None, "highlighted_evidence": ["Nowhere."]}]}]
    c = QasperCorpus.from_raw([paper])
    assert c.qrels() == {"q1": {}}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/corpus/test_qasper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.corpus'`

- [ ] **Step 3: Implement**

```python
# src/amrag/corpus/base.py
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    meta: dict


@dataclass(frozen=True)
class Query:
    qid: str
    text: str


class Corpus(Protocol):
    def documents(self) -> Iterator[Document]: ...
    def queries(self) -> Iterator[Query]: ...
    def qrels(self) -> dict[str, dict[str, int]]: ...
```

```python
# src/amrag/corpus/qasper.py
"""QASPER adapter. Retrieval unit = one evidence paragraph.

Evidence in QASPER is given as verbatim paragraph strings, so we resolve them
back to paragraph indices by exact match. Unresolvable evidence is dropped and
counted -- silently ignoring it would inflate Recall@k.
"""
from typing import Iterator

from amrag.corpus.base import Corpus, Document, Query


def _doc_id(paper_id: str, para_idx: int) -> str:
    return f"{paper_id}::{para_idx}"


def _paragraphs_from_paper(paper: dict) -> list[str]:
    out: list[str] = []
    for section in paper["full_text"]["paragraphs"]:
        out.extend(section)
    return out


class QasperCorpus(Corpus):
    def __init__(self, papers: list[dict]) -> None:
        self._papers = papers
        self.dropped_evidence = 0

    @classmethod
    def from_raw(cls, papers: list[dict]) -> "QasperCorpus":
        return cls(papers)

    @classmethod
    def load(cls, split: str = "test") -> "QasperCorpus":
        from datasets import load_dataset
        ds = load_dataset("allenai/qasper", split=split)
        return cls([dict(row) for row in ds])

    def documents(self) -> Iterator[Document]:
        for paper in self._papers:
            for i, para in enumerate(_paragraphs_from_paper(paper)):
                yield Document(doc_id=_doc_id(paper["id"], i), text=para,
                               meta={"paper_id": paper["id"]})

    def queries(self) -> Iterator[Query]:
        for paper in self._papers:
            qas = paper["qas"]
            for qid, question in zip(qas["question_id"], qas["question"]):
                yield Query(qid=qid, text=question)

    def qrels(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for paper in self._papers:
            paras = _paragraphs_from_paper(paper)
            index = {p: i for i, p in enumerate(paras)}
            qas = paper["qas"]
            for qid, ann in zip(qas["question_id"], qas["answers"]):
                rels: dict[str, int] = {}
                for a in ann["answer"]:
                    for ev in a["evidence"]:
                        if ev in index:
                            rels[_doc_id(paper["id"], index[ev])] = 1
                        else:
                            self.dropped_evidence += 1
                out[qid] = rels
        return out
```

Also create an empty `src/amrag/corpus/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/corpus/test_qasper.py -v`
Expected: 5 passed

- [ ] **Step 5: Report the dropped-evidence rate on real data**

This number matters. If it is large, our Recall@k ceiling is below 1.0 and every subsequent comparison is skewed.

```bash
source scripts/env.sh
python -c "
from amrag.corpus.qasper import QasperCorpus
c = QasperCorpus.load('test')
q = c.qrels()
kept = sum(len(v) for v in q.values())
print(f'queries={len(q)} kept_evidence={kept} dropped={c.dropped_evidence}')
print(f'queries with zero resolvable evidence: {sum(1 for v in q.values() if not v)}')
"
```

Record the output in the commit message. Expected: `queries` ≈ 1451 (a *derived* figure — the dataset card does not state the test split size; trust what prints, not the spec).

- [ ] **Step 6: Commit**

```bash
git add src/amrag/corpus tests/corpus
git commit -m "feat: QASPER corpus adapter with paragraph-level qrels

Dropped-evidence rate on test split: <paste Step 5 output>"
```

---

### Task 5: LitSearch adapter

**Files:**
- Create: `src/amrag/corpus/litsearch.py`, `tests/corpus/test_litsearch.py`

**Interfaces:**
- Consumes: `Document`, `Query`, `Corpus` from Task 4.
- Produces: `LitSearchCorpus`, same three methods. Retrieval unit = **one paper** (`corpus_clean`: title + abstract). qrels come from the query row's **`corpusids`** field.

> ⚠️ **Verified schema trap (2026-07-10).** The `query` config's relevance field is **`corpusids`**, *not* `citations`:
> `['query_set', 'query', 'specificity', 'quality', 'corpusids']` — 597 rows.
> The `corpus_clean` config **also has a field named `citations`** — but it means *that paper's own outgoing bibliography*
> (`['corpusid','title','abstract','citations','full_paper']`, 64,183 rows). Reading qrels from that field would
> yield plausible, entirely fictitious relevance labels. **Use `corpusids`, from the query rows, and nothing else.**

- [ ] **Step 1: Write the failing test**

```python
# tests/corpus/test_litsearch.py
from amrag.corpus.base import Document, Query
from amrag.corpus.litsearch import LitSearchCorpus

RAW_CORPUS = [{"corpusid": 101, "title": "Deep Nets", "abstract": "We study nets."}]
RAW_QUERIES = [{"query_set": "s", "query": "papers on nets?", "corpusids": [101], "specificity": 0}]

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
    c = LitSearchCorpus.from_raw(RAW_CORPUS, [{**RAW_QUERIES[0], "corpusids": [101, 999]}])
    assert c.qrels() == {"q0": {"101": 1}}
    assert c.dropped_citations == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/corpus/test_litsearch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.corpus.litsearch'`

- [ ] **Step 3: Implement**

```python
# src/amrag/corpus/litsearch.py
"""LitSearch adapter. Retrieval unit = one paper (title + abstract).

Uses the `corpus_clean` config (1.26 GB). `corpus_s2orc` (full text, 1.50 GB)
is deliberately not loaded in M1 -- see the spec's disk budget.
"""
from typing import Iterator

from amrag.corpus.base import Corpus, Document, Query


class LitSearchCorpus(Corpus):
    def __init__(self, corpus: list[dict], queries: list[dict]) -> None:
        self._corpus = corpus
        self._queries = queries
        self.dropped_citations = 0

    @classmethod
    def from_raw(cls, corpus: list[dict], queries: list[dict]) -> "LitSearchCorpus":
        return cls(corpus, queries)

    @classmethod
    def load(cls) -> "LitSearchCorpus":
        from datasets import load_dataset
        corpus = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
        queries = load_dataset("princeton-nlp/LitSearch", "query", split="full")
        return cls([dict(r) for r in corpus], [dict(r) for r in queries])

    def documents(self) -> Iterator[Document]:
        for row in self._corpus:
            yield Document(
                doc_id=str(row["corpusid"]),
                text=f"{row['title']}\n\n{row['abstract']}",
                meta={"title": row["title"]},
            )

    def queries(self) -> Iterator[Query]:
        for i, row in enumerate(self._queries):
            yield Query(qid=f"q{i}", text=row["query"])

    def qrels(self) -> dict[str, dict[str, int]]:
        """Relevance comes from the query row's `corpusids`.

        NOT from the corpus row's `citations` field, which exists but means
        that paper's own outgoing bibliography. Counters accumulate in a local
        and are assigned once, so repeated calls report identical numbers.
        """
        known = {str(r["corpusid"]) for r in self._corpus}
        out: dict[str, dict[str, int]] = {}
        dropped = 0
        for i, row in enumerate(self._queries):
            rels: dict[str, int] = {}
            for cid in row["corpusids"]:
                if str(cid) in known:
                    rels[str(cid)] = 1
                else:
                    dropped += 1
            out[f"q{i}"] = rels
        self.dropped_citations = dropped
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/corpus/test_litsearch.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/corpus/litsearch.py tests/corpus/test_litsearch.py
git commit -m "feat: LitSearch corpus adapter (corpus_clean, paper-level qrels)"
```

---

### Task 6: BM25 sparse retriever

**Files:**
- Create: `src/amrag/index/__init__.py`, `src/amrag/index/text.py`, `tests/index/test_bm25.py`

**Interfaces:**
- Consumes: `Hit` (Task 2), `Document` (Task 4).
- Produces: `BM25Retriever.build(docs: Iterable[Document]) -> BM25Retriever`, `.retrieve(query, k) -> list[Hit]`. Emits `granularity="passage"`, `modality="text"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/index/test_bm25.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/index/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.index'`

- [ ] **Step 3: Implement**

```python
# src/amrag/index/text.py
"""Text retrievers: BM25 (lexical) and dense (semantic).

Deck slide 25: dense captures paraphrase, sparse captures exact terms, IDs and
rare phrases. On a paper corpus the sparse arm is what finds "ColBERT" or
"nDCG@10"; the dense arm is what finds "late interaction retrieval".
"""
from typing import Iterable

from rank_bm25 import BM25Okapi

from amrag.corpus.base import Document
from amrag.types import Hit


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, doc_ids: list[str], bm25: BM25Okapi) -> None:
        self._doc_ids = doc_ids
        self._bm25 = bm25

    @classmethod
    def build(cls, docs: Iterable[Document]) -> "BM25Retriever":
        docs = list(docs)
        if not docs:
            raise ValueError("cannot build BM25 index over zero documents")
        return cls([d.doc_id for d in docs], BM25Okapi([_tokenize(d.text) for d in docs]))

    def retrieve(self, query: str, k: int) -> list[Hit]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Hit(doc_id=self._doc_ids[i], score=float(scores[i]),
                granularity="passage", modality="text")
            for i in order
        ]
```

Also create an empty `src/amrag/index/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/index/test_bm25.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/index tests/index
git commit -m "feat: BM25 sparse retriever"
```

---

### Task 7: Dense retriever (BGE-M3), with an injectable encoder

Do not make the unit tests download 2.29 GB of weights. The encoder is a constructor argument; tests pass a fake.

**Files:**
- Modify: `src/amrag/index/text.py`
- Create: `tests/index/test_dense.py`

**Interfaces:**
- Consumes: `Hit`, `Document`.
- Produces: `Encoder` protocol with `encode(texts: list[str]) -> np.ndarray` (shape `(n, dim)`, **L2-normalised**); `DenseRetriever.build(docs, encoder)`, `.retrieve(query, k)`; `BGEM3Encoder()` implementing `Encoder`.

- [ ] **Step 1: Write the failing test**

```python
# tests/index/test_dense.py
import numpy as np
import pytest
from amrag.corpus.base import Document
from amrag.index.text import DenseRetriever

DOCS = [Document("d1", "cat", {}), Document("d2", "photon", {})]

class FakeEncoder:
    """d1 -> [1,0], d2 -> [0,1]; query 'cat' -> [1,0]."""
    TABLE = {"cat": [1.0, 0.0], "photon": [0.0, 1.0]}
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.TABLE.get(t, [0.0, 0.0]) for t in texts], dtype=np.float32)

def test_retrieves_nearest_neighbour_by_cosine():
    r = DenseRetriever.build(DOCS, FakeEncoder())
    hits = r.retrieve("cat", k=1)
    assert hits[0].doc_id == "d1"
    assert hits[0].score == pytest.approx(1.0)

def test_declares_passage_granularity_and_text_modality():
    r = DenseRetriever.build(DOCS, FakeEncoder())
    h = r.retrieve("cat", k=1)[0]
    assert (h.granularity, h.modality) == ("passage", "text")

def test_orders_all_docs_when_k_exceeds_corpus():
    r = DenseRetriever.build(DOCS, FakeEncoder())
    hits = r.retrieve("cat", k=10)
    assert [h.doc_id for h in hits] == ["d1", "d2"]

@pytest.mark.slow
def test_bge_m3_encoder_produces_normalised_vectors():
    from amrag.index.text import BGEM3Encoder
    v = BGEM3Encoder().encode(["hello world"])
    assert v.shape[1] == 1024
    assert np.linalg.norm(v[0]) == pytest.approx(1.0, abs=1e-3)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/index/test_dense.py -v`
Expected: FAIL — `ImportError: cannot import name 'DenseRetriever'`

- [ ] **Step 3: Implement (append to `src/amrag/index/text.py`)**

```python
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised embeddings."""
        ...


class BGEM3Encoder:
    """BAAI/bge-m3 dense head. 1024-dim, L2-normalised.

    Repo ships a duplicate ONNX export; `sentence-transformers` pulls only the
    PyTorch weights (~2.29 GB of the 4.57 GB repo).
    """
    def __init__(self, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer("BAAI/bge-m3", device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._m.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=16, show_progress_bar=False,
        ).astype(np.float32)


class DenseRetriever:
    def __init__(self, doc_ids: list[str], matrix: np.ndarray, encoder: Encoder) -> None:
        self._doc_ids = doc_ids
        self._matrix = matrix          # (n, dim), L2-normalised
        self._encoder = encoder

    @classmethod
    def build(cls, docs: Iterable[Document], encoder: Encoder) -> "DenseRetriever":
        docs = list(docs)
        if not docs:
            raise ValueError("cannot build dense index over zero documents")
        matrix = encoder.encode([d.text for d in docs])
        return cls([d.doc_id for d in docs], matrix, encoder)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        q = self._encoder.encode([query])[0]
        scores = self._matrix @ q      # cosine, inputs are normalised
        order = np.argsort(-scores)[:k]
        return [
            Hit(doc_id=self._doc_ids[i], score=float(scores[i]),
                granularity="passage", modality="text")
            for i in order
        ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/index/test_dense.py -v`
Expected: 3 passed, 1 skipped (the `slow` marker)

Then, once, with weights: `python -m pytest tests/index/test_dense.py -v -m slow`
Expected: 1 passed. **If dim ≠ 1024, stop and fix the assertion to reality.**

- [ ] **Step 5: Commit**

```bash
git add src/amrag/index/text.py tests/index/test_dense.py
git commit -m "feat: dense retriever with injectable encoder; BGE-M3 impl"
```

---

### Task 8: Reciprocal Rank Fusion

**Files:**
- Create: `src/amrag/retrieve/__init__.py`, `src/amrag/retrieve/fuse.py`, `tests/retrieve/test_fuse.py`

**Interfaces:**
- Consumes: `Hit`.
- Produces: `rrf_fuse(runs: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]`. Fused `score` is the RRF score. `granularity`/`modality` are inherited from the **best-ranked** contributing hit — never invented.

- [ ] **Step 1: Write the failing test**

Fixture values computed by execution:

```python
# tests/retrieve/test_fuse.py
import pytest
from amrag.retrieve.fuse import rrf_fuse
from amrag.types import Hit

def h(doc_id, score=0.0, granularity="passage", modality="text"):
    return Hit(doc_id=doc_id, score=score, granularity=granularity, modality=modality)

RUN_A = [h("d1"), h("d2")]
RUN_B = [h("d2"), h("d3")]

def test_doc_ranked_well_in_both_runs_wins():
    fused = rrf_fuse([RUN_A, RUN_B], k=3)
    assert [x.doc_id for x in fused] == ["d2", "d1", "d3"]

def test_rrf_scores_match_the_formula():
    fused = rrf_fuse([RUN_A, RUN_B], k=3)
    by_id = {x.doc_id: x.score for x in fused}
    assert by_id["d2"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-9)   # 0.0325225
    assert by_id["d1"] == pytest.approx(1 / 61, abs=1e-9)            # 0.0163934
    assert by_id["d3"] == pytest.approx(1 / 62, abs=1e-9)            # 0.0161290

def test_granularity_is_inherited_not_invented():
    visual = [h("d9", granularity="page", modality="visual")]
    fused = rrf_fuse([visual], k=1)
    assert (fused[0].granularity, fused[0].modality) == ("page", "visual")

def test_mixed_modality_inherits_from_best_ranked_contributor():
    text_run = [h("d1", granularity="passage", modality="text")]
    vis_run = [h("d0", granularity="page", modality="visual"), h("d1", granularity="page", modality="visual")]
    fused = rrf_fuse([text_run, vis_run], k=2)
    d1 = next(x for x in fused if x.doc_id == "d1")
    assert d1.granularity == "passage"   # rank 1 in text_run beats rank 2 in vis_run

def test_empty_runs_yield_empty_result():
    assert rrf_fuse([], k=5) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/retrieve/test_fuse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.retrieve'`

- [ ] **Step 3: Implement**

```python
# src/amrag/retrieve/fuse.py
"""Reciprocal Rank Fusion (deck slide 28).

RRF combines runs without needing their scores to be commensurable -- it uses
only ranks. rrf_k=60 is the value from Cormack et al. (2009).
"""
from amrag.types import Hit


def rrf_fuse(runs: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, Hit]] = {}   # doc_id -> (best_rank, hit)

    for run in runs:
        for rank, hit in enumerate(run, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if hit.doc_id not in best or rank < best[hit.doc_id][0]:
                best[hit.doc_id] = (rank, hit)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out: list[Hit] = []
    for doc_id, score in ordered:
        src = best[doc_id][1]
        out.append(Hit(doc_id=doc_id, score=score, granularity=src.granularity,
                       modality=src.modality, page=src.page, span=src.span))
    return out
```

Also create an empty `src/amrag/retrieve/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/retrieve/test_fuse.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/retrieve tests/retrieve
git commit -m "feat: RRF fusion preserving honest granularity"
```

---

### Task 9: Cross-encoder reranker

**Files:**
- Create: `src/amrag/retrieve/rerank.py`, `tests/retrieve/test_rerank.py`

**Interfaces:**
- Consumes: `Hit`, `Document`.
- Produces: `Scorer` protocol with `score(pairs: list[tuple[str, str]]) -> list[float]`; `rerank(query, hits, doc_texts: dict[str,str], scorer, k) -> list[Hit]`; `BGEReranker()` implementing `Scorer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieve/test_rerank.py
import pytest
from amrag.retrieve.rerank import rerank
from amrag.types import Hit

TEXTS = {"d1": "irrelevant filler", "d2": "the answer is 42"}
HITS = [Hit("d1", 9.0, "passage", "text"), Hit("d2", 1.0, "passage", "text")]

class FakeScorer:
    def score(self, pairs): return [0.1 if "filler" in d else 0.9 for _, d in pairs]

def test_reranker_overrides_first_stage_order():
    out = rerank("what is the answer", HITS, TEXTS, FakeScorer(), k=2)
    assert [h.doc_id for h in out] == ["d2", "d1"]

def test_scores_are_replaced_by_reranker_scores():
    out = rerank("q", HITS, TEXTS, FakeScorer(), k=2)
    assert out[0].score == pytest.approx(0.9)

def test_granularity_and_modality_survive_reranking():
    out = rerank("q", HITS, TEXTS, FakeScorer(), k=1)
    assert (out[0].granularity, out[0].modality) == ("passage", "text")

def test_k_truncates_after_reordering():
    out = rerank("q", HITS, TEXTS, FakeScorer(), k=1)
    assert [h.doc_id for h in out] == ["d2"]

def test_hit_missing_from_doc_texts_raises_rather_than_scoring_empty_string():
    with pytest.raises(KeyError):
        rerank("q", [Hit("dX", 1.0, "passage", "text")], TEXTS, FakeScorer(), k=1)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/retrieve/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.retrieve.rerank'`

- [ ] **Step 3: Implement**

```python
# src/amrag/retrieve/rerank.py
"""Cross-encoder reranking (deck slide 28): score (query, chunk) pairs jointly
rather than trusting bi-encoder cosine similarity.
"""
from typing import Protocol

from amrag.types import Hit


class Scorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class BGEReranker:
    def __init__(self, device: str = "cuda") -> None:
        from FlagEmbedding import FlagReranker
        self._m = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device=device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._m.compute_score(pairs, normalize=True)
        return [scores] if isinstance(scores, float) else list(scores)


def rerank(query: str, hits: list[Hit], doc_texts: dict[str, str],
           scorer: Scorer, k: int) -> list[Hit]:
    if not hits:
        return []
    pairs = [(query, doc_texts[h.doc_id]) for h in hits]   # KeyError is intentional
    scores = scorer.score(pairs)
    rescored = [
        Hit(doc_id=h.doc_id, score=float(s), granularity=h.granularity,
            modality=h.modality, page=h.page, span=h.span)
        for h, s in zip(hits, scores)
    ]
    rescored.sort(key=lambda h: -h.score)
    return rescored[:k]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/retrieve/test_rerank.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/retrieve/rerank.py tests/retrieve/test_rerank.py
git commit -m "feat: cross-encoder reranker with injectable scorer"
```

---

### Task 10: HyDE query transformation

**Files:**
- Create: `src/amrag/retrieve/hyde.py`, `tests/retrieve/test_hyde.py`

**Interfaces:**
- Consumes: `LLM` protocol (defined here, implemented in Task 11) with `complete(prompt: str) -> str`.
- Produces: `hyde_transform(query: str, llm: LLM) -> str` returning `f"{query}\n\n{hypothetical}"`.

The concatenation matters: HyDE's hypothetical document can drift off-intent. Keeping the original query anchors it. Deck slide 27: *"The goal is not to change the user's intent."*

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieve/test_hyde.py
from amrag.retrieve.hyde import HYDE_PROMPT, hyde_transform

class FakeLLM:
    def __init__(self, reply="Nets are trained with backprop."):
        self.reply, self.seen = reply, []
    def complete(self, prompt: str) -> str:
        self.seen.append(prompt); return self.reply

def test_hypothetical_answer_is_appended_to_original_query():
    llm = FakeLLM()
    assert hyde_transform("how are nets trained?", llm) == (
        "how are nets trained?\n\nNets are trained with backprop."
    )

def test_prompt_embeds_the_user_query():
    llm = FakeLLM()
    hyde_transform("how are nets trained?", llm)
    assert "how are nets trained?" in llm.seen[0]
    assert HYDE_PROMPT.split("{query}")[0] in llm.seen[0]

def test_empty_llm_reply_falls_back_to_the_bare_query():
    assert hyde_transform("q", FakeLLM(reply="   ")) == "q"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/retrieve/test_hyde.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.retrieve.hyde'`

- [ ] **Step 3: Implement**

```python
# src/amrag/retrieve/hyde.py
"""HyDE (deck slide 27): generate a hypothetical answer and embed *that*,
on the theory that an answer looks more like the target passage than the
question does.

We concatenate rather than replace: a hallucinated hypothetical can drift off
the user's intent, and the original query anchors it.
"""
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


HYDE_PROMPT = (
    "Write a short passage from a scientific paper that would directly answer "
    "the following question. Do not preface it. Two or three sentences.\n\n"
    "Question: {query}\n\nPassage:"
)


def hyde_transform(query: str, llm: LLM) -> str:
    hypothetical = llm.complete(HYDE_PROMPT.format(query=query)).strip()
    if not hypothetical:
        return query
    return f"{query}\n\n{hypothetical}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/retrieve/test_hyde.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/retrieve/hyde.py tests/retrieve/test_hyde.py
git commit -m "feat: HyDE query transform anchored to original query"
```

---

### Task 11: DeepSeek client and the grounded prompt

**Files:**
- Create: `src/amrag/generate/__init__.py`, `src/amrag/generate/llm.py`, `src/amrag/generate/prompt.py`, `tests/generate/test_prompt.py`, `tests/generate/test_llm.py`

**Interfaces:**
- Consumes: `LLM` protocol (Task 10), `Hit` (Task 2).
- Produces: `DeepSeekLLM(model="deepseek-v4-flash")` implementing `complete`; `build_grounded_prompt(query, hits, doc_texts) -> str`; constant `INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"`.

Deck slide 35 is the requirement: *"It is better to say 'not enough evidence' than to generate an unsupported answer."* The prompt must make abstention a first-class output, or M4's rejection metric has nothing to measure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/generate/test_prompt.py
from amrag.generate.prompt import INSUFFICIENT_EVIDENCE, build_grounded_prompt
from amrag.types import Hit

TEXTS = {"d1": "Backprop trains nets.", "d2": "Photons are bosons."}
HITS = [Hit("d1", 0.9, "passage", "text"), Hit("d2", 0.4, "passage", "text")]

def test_every_hit_appears_with_a_numbered_citation_tag():
    p = build_grounded_prompt("how are nets trained?", HITS, TEXTS)
    assert "[1]" in p and "Backprop trains nets." in p
    assert "[2]" in p and "Photons are bosons." in p

def test_prompt_states_the_abstention_contract():
    p = build_grounded_prompt("q", HITS, TEXTS)
    assert INSUFFICIENT_EVIDENCE in p

def test_prompt_carries_the_user_query():
    assert "how are nets trained?" in build_grounded_prompt("how are nets trained?", HITS, TEXTS)

def test_zero_hits_still_produces_a_prompt_that_permits_abstention():
    p = build_grounded_prompt("q", [], {})
    assert INSUFFICIENT_EVIDENCE in p
```

```python
# tests/generate/test_llm.py
import pytest
from amrag.generate.llm import DeepSeekLLM

class FakeChoice:
    def __init__(self, content): self.message = type("M", (), {"content": content})()

class FakeClient:
    def __init__(self): self.kwargs = None; self.chat = type("C", (), {"completions": self})()
    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("R", (), {"choices": [FakeChoice("hi")]})()

def test_complete_returns_message_content():
    c = FakeClient()
    assert DeepSeekLLM(client=c).complete("hello") == "hi"

def test_uses_the_pinned_judge_model_by_default():
    c = FakeClient()
    DeepSeekLLM(client=c).complete("hello")
    assert c.kwargs["model"] == "deepseek-v4-flash"

def test_temperature_is_zero_for_reproducible_evaluation():
    c = FakeClient()
    DeepSeekLLM(client=c).complete("hello")
    assert c.kwargs["temperature"] == 0.0

def test_reasoner_model_is_rejected_because_it_exposes_no_logprobs():
    with pytest.raises(ValueError, match="logprobs"):
        DeepSeekLLM(client=FakeClient(), model="deepseek-reasoner")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/generate -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.generate'`

- [ ] **Step 3: Implement**

```python
# src/amrag/generate/prompt.py
"""Grounded prompt construction (deck slides 33-35)."""
from amrag.types import Hit

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

_TEMPLATE = """\
Answer the question using ONLY the numbered evidence below.

Rules:
- Cite every claim with the bracketed number of the evidence supporting it, e.g. [1].
- Do not use knowledge that is absent from the evidence.
- If the evidence does not contain enough information to answer, reply with
  exactly {sentinel} and nothing else. Saying {sentinel} is always preferable
  to an unsupported answer.

Evidence:
{evidence}

Question: {query}

Answer:"""


def build_grounded_prompt(query: str, hits: list[Hit], doc_texts: dict[str, str]) -> str:
    if hits:
        evidence = "\n".join(
            f"[{i}] {doc_texts[h.doc_id]}" for i, h in enumerate(hits, start=1)
        )
    else:
        evidence = "(no evidence retrieved)"
    return _TEMPLATE.format(sentinel=INSUFFICIENT_EVIDENCE, evidence=evidence, query=query)
```

```python
# src/amrag/generate/llm.py
"""DeepSeek text client.

Pinned to deepseek-v4-flash: it is the fixed judge for the whole project, and
it exposes logprobs/top_logprobs (which `deepseek-reasoner` does not). M2's
L3Score depends on that, so the reasoner model is rejected at construction.

DeepSeek serves no vision model -- M2's generator is Qwen3-VL, not this class.
"""
import os

DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekLLM:
    def __init__(self, client=None, model: str = DEFAULT_MODEL) -> None:
        if "reasoner" in model:
            raise ValueError(
                f"{model!r} exposes no logprobs; L3Score requires them. Use {DEFAULT_MODEL!r}."
            )
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        self._client = client
        self._model = model

    def complete(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return r.choices[0].message.content
```

Also create an empty `src/amrag/generate/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/generate -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/generate tests/generate
git commit -m "feat: DeepSeek client (judge-pinned) and grounded prompt with abstention"
```

---

### Task 12: Vendor the official QASPER evaluator

**Do not reimplement Answer-F1 or Evidence-F1.** Their exact definitions (annotator maximisation, normalisation, unanswerable handling) are subtle, and a reimplementation produces numbers that cannot be compared to any published result.

**Files:**
- Create: `vendor/__init__.py`, `vendor/qasper_eval.py` (downloaded, unmodified), `src/amrag/eval/answer.py`, `tests/eval/test_answer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `score_answers(predictions: dict[str, str], gold: dict) -> dict[str, float]` returning at least `{"answer_f1": ..., "evidence_f1": ...}`.

- [ ] **Step 1: Fetch the upstream evaluator**

```bash
mkdir -p vendor && touch vendor/__init__.py
curl -fsSL -o vendor/qasper_eval.py \
  https://raw.githubusercontent.com/allenai/qasper-led-baseline/main/scripts/evaluator.py
head -30 vendor/qasper_eval.py    # confirm it is the scorer, not an HTML 404 page
```

**Verified 2026-07-10 (HTTP 200).** The file exposes exactly the symbols we need:

| Symbol | What it is |
|---|---|
| `normalize_answer(s)` | SQuAD-style normalisation: lowercase, strip punctuation/articles/extra whitespace |
| `token_f1_score(prediction, ground_truth)` | **Answer-F1** (token-level). Returns integer `0` — not `0.0` — when there is no overlap |
| `paragraph_f1_score(prediction, ground_truth)` | **Evidence-F1** (set-F1 over evidence paragraph strings) |

> ⚠️ **Use `paragraph_f1_score`. Do not hand-roll set-F1.** Upstream contains a special case that a
> naive implementation gets wrong:
> ```python
> if not ground_truth and not prediction:
>     return 1.0     # unanswerable question + empty prediction == perfect
> ```
> A hand-rolled version returning `0.0` there would silently penalise correct abstention — on QASPER's
> test split, 142 queries have no resolvable evidence, so this is not a hypothetical edge case.

If the URL 404s, locate the current path in `allenai/qasper-led-baseline` and record the exact commit SHA in the commit message. **Never hand-write a substitute.**

- [ ] **Step 2: Write the failing test**

```python
# tests/eval/test_answer.py
import pytest
from amrag.eval.answer import build_gold, score_answers

GOLD = {
    "q1": {"answers": ["backpropagation"], "evidence": ["Nets are trained with backprop."]},
}

def test_exact_match_answer_scores_one():
    s = score_answers({"q1": "backpropagation"}, GOLD)
    assert s["answer_f1"] == pytest.approx(1.0)

def test_completely_wrong_answer_scores_zero():
    s = score_answers({"q1": "photons"}, GOLD)
    assert s["answer_f1"] == pytest.approx(0.0)

def test_partial_token_overlap_scores_between_zero_and_one():
    s = score_answers({"q1": "backpropagation and photons"}, GOLD)
    assert 0.0 < s["answer_f1"] < 1.0

def test_missing_prediction_is_scored_as_empty_not_skipped():
    """A question we failed to answer must count against us, not vanish."""
    s = score_answers({}, GOLD)
    assert s["answer_f1"] == pytest.approx(0.0)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/eval/test_answer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.eval.answer'`

- [ ] **Step 4: Implement the thin adapter**

```python
# src/amrag/eval/answer.py
"""Adapter onto the vendored official QASPER evaluator.

The scoring functions live in vendor/qasper_eval.py, downloaded unmodified from
allenai/qasper-led-baseline. We only reshape our data into its expected form.
Reimplementing these metrics would silently break comparability with the paper.
"""
from vendor.qasper_eval import token_f1_score


def score_answers(predictions: dict[str, str], gold: dict) -> dict[str, float]:
    """`gold` maps qid -> {"answers": [str, ...], "evidence": [str, ...]}.

    A qid present in `gold` but absent from `predictions` scores 0 -- an
    unanswered question is a failure, not an exemption.
    """
    if not gold:
        return {"answer_f1": 0.0, "evidence_f1": 0.0}

    answer_scores = []
    for qid, g in gold.items():
        pred = predictions.get(qid, "")
        answer_scores.append(max(token_f1_score(pred, ref) for ref in g["answers"]))

    return {
        "answer_f1": sum(answer_scores) / len(answer_scores),
        "evidence_f1": 0.0,   # populated in Step 6
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/eval/test_answer.py -v`
Expected: 4 passed.

If `token_f1_score` is not the upstream symbol name, read `vendor/qasper_eval.py` and use the real one. **Adjust our adapter, never the vendored file.**

- [ ] **Step 6: Add evidence scoring, driven by a new failing test**

Append to `tests/eval/test_answer.py`:

```python
def test_evidence_f1_rewards_retrieving_the_gold_paragraph():
    s = score_answers({"q1": "backpropagation"}, GOLD,
                      retrieved={"q1": ["Nets are trained with backprop."]})
    assert s["evidence_f1"] == pytest.approx(1.0)

def test_evidence_f1_punishes_retrieving_noise():
    s = score_answers({"q1": "backpropagation"}, GOLD,
                      retrieved={"q1": ["Nets are trained with backprop.", "Photons are bosons."]})
    assert s["evidence_f1"] == pytest.approx(2 / 3)   # P=1/2, R=1 -> F1=2/3

def test_evidence_f1_is_zero_when_nothing_retrieved():
    s = score_answers({"q1": "backpropagation"}, GOLD, retrieved={"q1": []})
    assert s["evidence_f1"] == 0.0
```

Run: `python -m pytest tests/eval/test_answer.py -v`
Expected: FAIL — `TypeError: score_answers() got an unexpected keyword argument 'retrieved'`

Now replace `src/amrag/eval/answer.py` with:

> ⚠️ **Do not write a scoring loop. Call upstream's `evaluate()`.**
> Reading `vendor/qasper_eval.py` reveals that a hand-written adapter diverges from the official
> protocol in four ways, each of which silently breaks comparability with the published numbers:
>
> | | Official `evaluate()` | A naive adapter |
> |---|---|---|
> | Gold shape | **per-annotator list** of `{answer, evidence, type}` | one flattened union |
> | Evidence-F1 | `max(paragraph_f1_score(pred, ref["evidence"]) for ref in refs)` | one call on the union — **systematically under-reports** whenever annotators disagree |
> | Unanswerable gold | the literal string `"Unanswerable"` | `""` — so correct abstention scores Answer-F1 **0**, not 1 |
> | Answer priority | `extractive_spans` (`", ".join`) → `free_form_answer` → `yes_no` (`"Yes"`/`"No"`) | some other order |
>
> Upstream also has a `text_evidence_only` flag that filters `"FLOAT SELECTED"` evidence — the
> official handling of the 459 figure/table annotations measured in Task 4.
>
> Our job is **shape translation only**: HuggingFace serves `qas` columnar (dict-of-lists);
> upstream expects it row-wise (list-of-dicts).

```python
# src/amrag/eval/answer.py
"""Thin shape-adapter onto the vendored official QASPER evaluator.

We do NOT score anything ourselves. `vendor/qasper_eval.py` is byte-identical to
allenai/qasper-led-baseline, and its `get_answers_and_evidence()` + `evaluate()`
define the metrics the paper reports. We only translate HuggingFace's columnar
`qas` layout into the row-wise layout upstream expects.

Every scoring subtlety -- max-over-annotators, the "Unanswerable" gold string,
the extractive/abstractive/boolean priority, the empty-gold-empty-prediction
special case -- lives upstream and must stay there.
"""
from vendor.qasper_eval import evaluate, get_answers_and_evidence


def _rowwise(paper: dict) -> dict:
    """HuggingFace gives `qas` as a dict-of-lists; upstream wants a list-of-dicts."""
    qas = paper["qas"]
    return {
        "qas": [
            {"question_id": qid, "question": q, "answers": ann["answer"]}
            for qid, q, ann in zip(qas["question_id"], qas["question"], qas["answers"])
        ]
    }


def build_gold(papers: list[dict], text_evidence_only: bool = True) -> dict:
    """qid -> [{answer, evidence, type}, ...], one entry per annotator.

    `text_evidence_only=True` drops "FLOAT SELECTED" (figure/table) evidence,
    which a text-only retriever cannot reach. That is upstream's own flag, and
    it is what makes the Recall@k ceiling honest rather than punitive.
    """
    return get_answers_and_evidence({p["id"]: _rowwise(p) for p in papers}, text_evidence_only)


def score_answers(predictions: dict[str, dict], gold: dict) -> dict:
    """`predictions` maps qid -> {"answer": str, "evidence": [str, ...]}.

    Returns upstream's dict: "Answer F1", "Answer F1 by type", "Evidence F1",
    "Missing predictions". A qid in `gold` but absent from `predictions` scores
    0 on both metrics -- upstream counts it, it is not an exemption.
    """
    return evaluate(gold, predictions)
```

Re-run: `python -m pytest tests/eval/test_answer.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add vendor src/amrag/eval/answer.py tests/eval/test_answer.py
git commit -m "feat: vendored official QASPER evaluator + adapter

Upstream: allenai/qasper-led-baseline @ <SHA>. Not reimplemented."
```

---

### Task 13: The ablation harness

**Files:**
- Create: `src/amrag/eval/ablate.py`, `tests/eval/test_ablate.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Config(name:str, sparse:bool, dense:bool, rerank:bool, hyde:bool)`; `ABLATION_LADDER: list[Config]`; `to_markdown(rows: list[dict]) -> str`.

> An earlier draft of this line also promised `evaluate_config(cfg, corpus, retrievers, ks)`. There is
> no such function and there should not be: `scripts/run_m1.py` (Task 14) owns the evaluation loop, and
> a second entry point that nothing calls is dead weight. YAGNI.

`to_markdown` derives its columns from `rows[0]`. If a later row is missing one of those keys it must
raise `ValueError` naming the key — a `KeyError` from deep inside a format string, after a 40-minute
retrieval run, is a bad way to learn that two rows disagreed.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_ablate.py
from amrag.eval.ablate import ABLATION_LADDER, Config, to_markdown

def test_ladder_is_strictly_cumulative():
    """Each rung adds exactly one component to the rung below it."""
    for prev, cur in zip(ABLATION_LADDER, ABLATION_LADDER[1:]):
        added = sum((getattr(cur, f) and not getattr(prev, f))
                    for f in ("sparse", "dense", "rerank", "hyde"))
        removed = sum((getattr(prev, f) and not getattr(cur, f))
                      for f in ("sparse", "dense", "rerank", "hyde"))
        assert (added, removed) == (1, 0), f"{prev.name} -> {cur.name}"

def test_naive_rung_is_dense_only():
    naive = ABLATION_LADDER[0]
    assert (naive.dense, naive.sparse, naive.rerank, naive.hyde) == (True, False, False, False)

def test_markdown_table_has_one_row_per_config():
    rows = [{"config": "naive", "recall@10": 0.5}, {"config": "+hybrid", "recall@10": 0.6}]
    md = to_markdown(rows)
    assert md.count("\n") == 3          # header, separator, 2 rows -> 3 newlines
    assert "| naive |" in md and "| +hybrid |" in md

def test_config_rejects_retrieving_with_neither_arm():
    import pytest
    with pytest.raises(ValueError):
        Config(name="x", sparse=False, dense=False, rerank=False, hyde=False)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/eval/test_ablate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'amrag.eval.ablate'`

- [ ] **Step 3: Implement**

```python
# src/amrag/eval/ablate.py
"""The ablation ladder (deck slides 10-11, 43).

Each rung adds exactly one component, so a delta in the table attributes to
exactly one design decision. That is the whole point -- a table where two things
change at once diagnoses nothing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str
    sparse: bool
    dense: bool
    rerank: bool
    hyde: bool

    def __post_init__(self) -> None:
        if not (self.sparse or self.dense):
            raise ValueError(f"{self.name}: needs at least one retrieval arm")


ABLATION_LADDER: list[Config] = [
    Config("naive",    sparse=False, dense=True, rerank=False, hyde=False),
    Config("+hybrid",  sparse=True,  dense=True, rerank=False, hyde=False),
    Config("+rerank",  sparse=True,  dense=True, rerank=True,  hyde=False),
    Config("+hyde",    sparse=True,  dense=True, rerank=True,  hyde=True),
]


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/eval/test_ablate.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/amrag/eval/ablate.py tests/eval/test_ablate.py
git commit -m "feat: strictly-cumulative ablation ladder + markdown table"
```

---

### Task 14: Run M1 and record the result

**Files:**
- Create: `scripts/run_m1.py`, `results/m1_qasper.md`, `results/m1_litsearch.md`, `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: two markdown tables and a README that states what they mean.

- [ ] **Step 1: Write the driver**

```python
# scripts/run_m1.py
"""Run the ablation ladder on QASPER and LitSearch. Writes results/*.md.

The +hyde rung costs one LLM call per query. It is SKIPPED unless --with-hyde is
passed, and the skip is printed -- a silently-omitted rung that prints numbers
identical to the rung below it would read as "HyDE had no effect", which is a
fabricated finding.
"""
import argparse, pathlib

from amrag.corpus.litsearch import LitSearchCorpus
from amrag.corpus.qasper import QasperCorpus
from amrag.eval.ablate import ABLATION_LADDER, to_markdown
from amrag.eval.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from amrag.generate.llm import DeepSeekLLM
from amrag.index.text import BGEM3Encoder, BM25Retriever, DenseRetriever
from amrag.retrieve.fuse import rrf_fuse
from amrag.retrieve.hyde import hyde_transform
from amrag.retrieve.rerank import BGEReranker, rerank


def run(corpus, k: int = 10, limit: int = 0, with_hyde: bool = False) -> str:
    docs = list(corpus.documents())
    texts = {d.doc_id: d.text for d in docs}
    qrels = corpus.qrels()
    queries = list(corpus.queries())
    if limit:
        queries = queries[:limit]

    encoder, reranker = BGEM3Encoder(), BGEReranker()
    dense = DenseRetriever.build(docs, encoder)
    sparse = BM25Retriever.build(docs)
    llm = DeepSeekLLM() if with_hyde else None

    ladder = [c for c in ABLATION_LADDER if with_hyde or not c.hyde]
    for c in ABLATION_LADDER:
        if c not in ladder:
            print(f"SKIPPING rung {c.name!r}: needs --with-hyde (costs API calls)", flush=True)

    rows = []
    for cfg in ladder:
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
    a = ap.parse_args()

    corpus = QasperCorpus.load("test") if a.dataset == "qasper" else LitSearchCorpus.load()
    md = run(corpus, limit=a.limit, with_hyde=a.with_hyde)
    suffix = "_hyde" if a.with_hyde else ""
    out = pathlib.Path(f"results/m1_{a.dataset}{suffix}.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(f"\nwrote {out}\n{md}")
```

Two subtleties baked in above, both worth understanding rather than copying:

1. **HyDE expands the *search* text but not the *rerank* text.** The cross-encoder must score relevance to the user's actual question; feeding it a hallucinated passage would let the reranker reward documents that match the hallucination.
2. **A skipped rung announces itself.** Silence would look like a null result.

- [ ] **Step 2: Smoke-run on QASPER with a tiny subset**

```bash
source scripts/env.sh && source .venv/bin/activate
python scripts/run_m1.py --dataset qasper --limit 20
```

Expected: four rows print, `recall@10` strictly increasing down the ladder — **or not.** If `+rerank` *lowers* recall@10, that is real and expected: reranking reorders within the candidate set, it cannot add documents. Recall@10 can drop when reranking truncates 50→10. Report it; do not "fix" it.

- [ ] **Step 3: Full runs**

```bash
python scripts/run_m1.py --dataset qasper
python scripts/run_m1.py --dataset litsearch    # ~64k docs; expect a long encode pass
```

Record wall-clock and peak VRAM (`nvidia-smi` in another shell) in the commit message. These numbers feed M2's planning.

- [ ] **Step 4: Write the README**

It must contain: the two tables; the dropped-evidence rate from Task 4 Step 5; a sentence stating that Recall@k is capped below 1.0 by unresolvable evidence; and the explicit note that **no answer generation has been evaluated yet** — M1 measures retrieval only, per the deck's *"evaluate retrieval before evaluating answers."*

- [ ] **Step 5: Commit and tag**

```bash
git add scripts/run_m1.py results README.md
git commit -m "feat: M1 ablation results on QASPER and LitSearch

Wall-clock: <...>  Peak VRAM: <...>"
git tag m1-complete
```

---

---

### Task 15: Generate answers and score them (Evidence-F1 → Answer-F1)

The spec's M1 exit criterion demands both. Tasks 1–14 measure retrieval only; nothing yet calls `score_answers`. This task closes that gap.

Order matters and is not cosmetic: **Evidence-F1 is reported before Answer-F1** (deck slide 39). An Answer-F1 computed on top of unmeasured retrieval cannot be diagnosed — you would not know whether a wrong answer came from missing evidence or from bad grounding.

**Files:**
- Create: `scripts/run_m1_answers.py`, `results/m1_qasper_answers.md`
- Modify: `src/amrag/corpus/qasper.py` (add `gold_answers()`)

**Interfaces:**
- Consumes: `QasperCorpus`, `build_grounded_prompt`, `DeepSeekLLM`, `score_answers`, `rrf_fuse`, `rerank`.
- Produces: `QasperCorpus.gold_answers() -> dict[str, dict]` mapping qid → `{"answers": [str], "evidence": [str]}` — exactly the shape `score_answers` expects.

- [ ] **Step 1: Write the failing test**

Append to `tests/corpus/test_qasper.py`:

```python
def test_gold_answers_shape_matches_score_answers_contract():
    c = QasperCorpus.from_raw([RAW_PAPER])
    gold = c.gold_answers()
    assert gold == {"q1": {"answers": ["beta"], "evidence": ["Alpha beta."]}}

def test_unanswerable_question_yields_empty_answer_string():
    paper = {**RAW_PAPER, "qas": {**RAW_PAPER["qas"]}}
    paper["qas"]["answers"] = [{"answer": [{"evidence": [], "free_form_answer": "",
                                            "extractive_spans": [], "unanswerable": True,
                                            "yes_no": None, "highlighted_evidence": []}]}]
    assert QasperCorpus.from_raw([paper]).gold_answers() == {"q1": {"answers": [""], "evidence": []}}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/corpus/test_qasper.py::test_gold_answers_shape_matches_score_answers_contract -v`
Expected: FAIL — `AttributeError: 'QasperCorpus' object has no attribute 'gold_answers'`

- [ ] **Step 3: Expose the raw papers; let upstream build the gold**

Do **not** hand-build a gold dict. `eval/answer.py::build_gold` (Task 12) delegates to upstream's
`get_answers_and_evidence()`, which already knows the per-annotator shape, the `"Unanswerable"`
sentinel, the extractive/abstractive/boolean priority, and the `text_evidence_only` filter.

`QasperCorpus` only needs to hand over its raw rows:

```python
    def raw_papers(self) -> list[dict]:
        """The untouched HuggingFace rows, for vendor/qasper_eval.py to consume.

        Deliberately not reshaped: `build_gold` owns the translation, and the
        scoring protocol lives upstream where it cannot drift from the paper.
        """
        return self._papers
```

Test: `QasperCorpus.from_raw([RAW_PAPER]).raw_papers() == [RAW_PAPER]` (identity, not a copy).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/corpus/test_qasper.py -v`
Expected: 7 passed

- [ ] **Step 5: Write the answer-scoring driver**

```python
# scripts/run_m1_answers.py
"""Retrieve -> generate -> score. Reports Evidence-F1 BEFORE Answer-F1.

Costs one LLM call per query. Use --limit while iterating; the full QASPER test
split is ~1,451 questions.
"""
import argparse, pathlib

from amrag.corpus.qasper import QasperCorpus
from amrag.eval.ablate import to_markdown
from amrag.eval.answer import build_gold, score_answers
from amrag.generate.llm import DeepSeekLLM
from amrag.generate.prompt import INSUFFICIENT_EVIDENCE, build_grounded_prompt
from amrag.index.text import BGEM3Encoder, BM25Retriever, DenseRetriever
from amrag.retrieve.fuse import rrf_fuse
from amrag.retrieve.rerank import BGEReranker, rerank

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()

corpus = QasperCorpus.load("test")
docs = list(corpus.documents())
texts = {d.doc_id: d.text for d in docs}
gold = build_gold(corpus.raw_papers(), text_evidence_only=True)
queries = list(corpus.queries())
if a.limit:
    queries = queries[: a.limit]
    gold = {q.qid: gold[q.qid] for q in queries}

dense = DenseRetriever.build(docs, BGEM3Encoder())
sparse = BM25Retriever.build(docs)
reranker, llm = BGEReranker(), DeepSeekLLM()

predictions, abstentions = {}, 0
for q in queries:
    hits = rrf_fuse([dense.retrieve(q.text, 100), sparse.retrieve(q.text, 100)], k=100)
    hits = rerank(q.text, hits[:50], texts, reranker, k=a.k)
    answer = llm.complete(build_grounded_prompt(q.text, hits, texts)).strip()
    if answer == INSUFFICIENT_EVIDENCE:
        abstentions += 1
        # Upstream's gold answer for an unanswerable question is the literal
        # string "Unanswerable". Emitting "" here would score Answer-F1 = 0 for
        # a CORRECT abstention -- token_f1_score has no empty-empty special case
        # (unlike paragraph_f1_score). Say what the benchmark expects.
        answer = "Unanswerable"
    predictions[q.qid] = {"answer": answer, "evidence": [texts[h.doc_id] for h in hits]}

s = score_answers(predictions, gold)
rows = [{
    "config": "+rerank",
    "evidence_f1": s["Evidence F1"],      # retrieval first (deck slide 39)
    "answer_f1": s["Answer F1"],
    "missing_predictions": s["Missing predictions"],
    "abstention_rate": abstentions / len(queries),
}]
md = to_markdown(rows)
pathlib.Path("results").mkdir(exist_ok=True)
pathlib.Path("results/m1_qasper_answers.md").write_text(md)
print(md)
print("\nAnswer F1 by type:", s["Answer F1 by type"])   # extractive/abstractive/boolean/none
```

- [ ] **Step 6: Smoke-run on 20 questions, then the full split**

```bash
source scripts/env.sh && source .venv/bin/activate
python scripts/run_m1_answers.py --limit 20     # confirm cost and sanity first
python scripts/run_m1_answers.py                # full split
```

**Read `abstention_rate` before you read `answer_f1`.** If it is ~0, the model is never saying `INSUFFICIENT_EVIDENCE` and the abstention contract in the prompt is not working — M4's rejection metric will be meaningless. If it is very high, retrieval is failing and Evidence-F1 will show it.

Record actual API spend here. It is the first real data point for the spec's unverified "tens of dollars" estimate.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_m1_answers.py src/amrag/corpus/qasper.py tests/corpus/test_qasper.py results
git commit -m "feat: QASPER answer generation + Evidence-F1/Answer-F1 scoring

Evidence-F1: <...>  Answer-F1: <...>  Abstention: <...>  API spend: <...>"
```

---

## M1 Exit Criteria

- [ ] `pytest` green (`-m 'not slow'` for CI; `-m slow` once locally).
- [ ] `results/m1_qasper.md` and `results/m1_litsearch.md` exist and are committed.
- [ ] `results/m1_qasper_answers.md` exists, with **Evidence-F1 reported before Answer-F1**.
- [ ] The dropped-evidence rate is measured and written into the README.
- [ ] The abstention rate is non-zero and non-degenerate (the prompt contract works).
- [ ] Each ladder rung's delta is attributable to exactly one component.
- [ ] Wall-clock, peak VRAM, and **actual API spend** recorded — M2 sizing and the spec's cost estimate depend on them.

**Not an exit criterion:** that the numbers go up. If `+hyde` hurts, that is a finding. If `+rerank` lowers Recall@10 (it can — reranking reorders a candidate set, it cannot enlarge it), that is arithmetic, not a bug.

## Next Plans (not written yet, deliberately)

- **M2** — visual path. Its Task 1 is *"measure vectors/page emitted by ColQwen2.5"*, resolving the 196-vs-768 discrepancy that the spec's disk budget rests on.
- **Gate** — oracle-router gap. Thresholds pre-committed in spec §7.
- **M3** — routers. **R2's feature set cannot be specified until M1/M2 produce oracle labels**, which is why this plan stops here.
- **M4** — RAGChecker diagnostics + abstention rate.
