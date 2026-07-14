"""Persistent, content-addressed embedding cache (Task 17).

Wraps an Encoder (BGE-M3, or any fake in tests) and persists its output to
disk, keyed by (model_id, sha256(text)), so re-running `scripts/run_m1.py`
-- after a crash, a code change in a later ablation rung, or just to
re-read the numbers -- does not pay the ~67-minute (QASPER) / ~211-minute
(LitSearch) CPU encode cost again.

On-disk format
--------------
One file per (model_id, text) pair::

    root / model_id / <sha256(text.encode("utf-8")).hexdigest()>.npz

npz, not the more obvious single matrix.npy + a JSON row index, because it
lets each cache ENTRY be one self-contained atomic unit: the vector and
the model_id it was produced under travel together in one file, written
with one `os.replace`. A monolithic matrix + index needs the writer to
keep two files in lockstep across a crash -- get the ordering wrong and a
SIGINT leaves the index pointing at rows that don't exist yet, which is
exactly the "misaligned row" failure this task's brief warns about.
Per-key files make that failure mode structurally impossible: a key is
either one complete, valid file or it isn't there, full stop -- there is
no shared row-indexed structure to desynchronise. The cost is many small
files (one per unique paragraph, tens of thousands for QASPER/LitSearch),
which is fine on the disk this cache lives under (see scripts/env.sh).

Cache key: sha256 of the UTF-8 text bytes, never the text itself, so
filenames are fixed-length and filesystem-safe regardless of what the
corpus text contains (newlines, slashes, unicode, arbitrary length).

model_id is part of the key twice over: it is both the containing
directory AND a field written inside every file, checked against on load.
The directory alone already makes cross-model contamination structurally
impossible (a lookup rooted at root/bge-m3/ can never read
root/other-model/...), but the brief calls for an explicit assert on
load, and it is cheap insurance against a future bug where two
CachedEncoder instances end up pointed at the same root. See `_load`'s
docstring for why a mismatch there raises instead of self-healing.

device is deliberately NOT part of the key. CPU and GPU produce slightly
different floats for the same input (non-deterministic kernel
reductions), but both are valid L2-normalised embeddings of the same text
under the same model -- keying on device would force a second full encode
of the exact corpus this cache exists to avoid re-encoding, to capture a
difference smaller than DenseRetriever's own _NORM_TOL.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from amrag.index.text import Encoder

logger = logging.getLogger(__name__)


class CachedEncoder:
    """Read-through, write-through cache in front of an `Encoder`.

    Satisfies the Encoder Protocol (`encode(texts) -> np.ndarray`), so it
    drops into `DenseRetriever.build(docs, CachedEncoder(...))` unchanged,
    and is also what `DenseRetriever` stores and later calls per-query in
    `.retrieve()`. See `encode()` for how the two call shapes are told
    apart for the write-back decision.
    """

    def __init__(self, inner: Encoder, model_id: str, root: Path) -> None:
        self._inner = inner
        self._model_id = model_id
        self._dir = Path(root) / model_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._dir / f"{key}.npz"

    def _load(self, path: Path) -> Optional[np.ndarray]:
        """Return the cached vector, or None if it isn't usable: file
        missing, or truncated/malformed npz -- the SIGINT-mid-write case
        property 5 asks for. Both are ordinary, expected misses; the
        caller self-heals by recomputing and rewriting.

        A well-formed file whose embedded model_id disagrees with this
        instance's is a *different* kind of failure. The directory
        scoping already makes that structurally impossible in normal
        operation, so if it ever happens something is actually wrong (a
        stray copy, or two encoders sharing a root) -- not a transient
        write failure. Self-healing that by silently overwriting would
        not fix the real bug, it would just flip-flop the entry between
        callers, so this raises instead of returning None.
        """
        try:
            with np.load(path, allow_pickle=False) as data:
                stored_model_id = str(data["model_id"])
                vector = np.array(data["vector"], dtype=np.float32)
        except (OSError, EOFError, zipfile.BadZipFile, ValueError, KeyError):
            return None
        if stored_model_id != self._model_id:
            raise ValueError(
                f"cache identity mismatch at {path}: file was written for "
                f"model_id={stored_model_id!r}, but this CachedEncoder is "
                f"model_id={self._model_id!r}. Refusing to serve it. Cache "
                f"paths are model_id-scoped, so this should be structurally "
                f"impossible -- treat it as a bug (e.g. two encoders "
                f"sharing a cache root), not a corrupt file."
            )
        return vector

    def _store(self, path: Path, vector: np.ndarray) -> None:
        """Write to a temp file in the same directory, then os.replace.

        A SIGINT mid-write leaves the temp file behind (harmless clutter,
        `_load` never opens it) and `path` itself either absent or holding
        the last complete write -- never a half-written file under the
        name a reader will open.
        """
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    vector=vector.astype(np.float32, copy=False),
                    model_id=np.array(self._model_id),
                )
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def encode(self, texts: list[str]) -> np.ndarray:
        # Resolve every unique text at most once: a disk hit, or (if
        # missing/corrupt) a slot to fill via a single batched inner.encode
        # call. Duplicates within `texts` reuse whichever slot their text
        # already resolved to (property 3): inner.encode is called once per
        # unique missing text, never once per occurrence.
        resolved: dict[str, np.ndarray] = {}
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for t in texts:
            if t in resolved or t in seen_missing:
                continue
            v = self._load(self._path(t))
            if v is None:
                seen_missing.add(t)
                missing_texts.append(t)
            else:
                resolved[t] = v

        if missing_texts:
            fresh = self._inner.encode(missing_texts)
            for t, v in zip(missing_texts, fresh):
                resolved[t] = np.asarray(v, dtype=np.float32)

        # Persist only what build() passes. DenseRetriever.build calls
        # encode() once with the whole corpus (many texts, property 2's
        # "order" test and this module's own tests rely on that shape).
        # DenseRetriever.retrieve calls it once per query with a single-
        # element list, and HyDE-expanded query text is different every
        # run -- writing every one-shot query string to disk would grow
        # the cache unboundedly with entries that will never be looked up
        # again. Per-key writes are already cheap (one small file, no
        # fsync, no whole-index rewrite), so this gate is about avoiding
        # that unbounded, useless growth, not about I/O cost. Reads are
        # never gated: a singleton call that happens to hit an
        # already-cached text (e.g. two identical non-HyDE queries) still
        # benefits from the cache, it just never causes a new write.
        if len(texts) > 1:
            for t in missing_texts:
                self._store(self._path(t), resolved[t])

        miss_count = sum(1 for t in texts if t in seen_missing)
        self.misses += miss_count
        self.hits += len(texts) - miss_count

        return np.stack([resolved[t] for t in texts]).astype(np.float32)
