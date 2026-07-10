"""Tests for CachedEncoder (Task 17).

Fakes + tmp_path only: no real model weights, no network, nothing written
outside pytest's tmp_path. See .superpowers/sdd/task17-brief.md for the six
required properties this file is organised around.
"""
import hashlib
import os
import shutil

import numpy as np
import pytest

from amrag.corpus.base import Document
from amrag.index.cache import CachedEncoder
from amrag.index.text import DenseRetriever


def _vec(text: str, dim: int = 4) -> np.ndarray:
    """Deterministic, L2-normalised, distinct-per-text unit vector."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(h[: dim * 4], dtype=np.uint32).astype(np.float64)
    raw = raw / np.linalg.norm(raw)
    return raw.astype(np.float32)


class SpyEncoder:
    """Records every call so tests can assert call counts and contents."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([_vec(t) for t in texts]).astype(np.float32)

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# Property 1: model_id is part of the cache identity.
# ---------------------------------------------------------------------------

def test_cache_path_is_scoped_by_model_id(tmp_path):
    enc_a = CachedEncoder(SpyEncoder(), model_id="model-a", root=tmp_path)
    enc_a.encode(["hello", "world"])
    model_a_dir = tmp_path / "model-a"
    assert model_a_dir.is_dir()
    assert list(model_a_dir.glob("*.npz")), "expected cache files under model-a/"
    assert not (tmp_path / "model-b").exists()


def test_different_model_ids_never_share_a_hit(tmp_path):
    """Same text, two model_ids: the second encoder must still be a miss --
    proof that vectors from one model are never served to a caller that
    asked for a different model."""
    spy_a = SpyEncoder()
    spy_b = SpyEncoder()
    enc_a = CachedEncoder(spy_a, model_id="model-a", root=tmp_path)
    enc_b = CachedEncoder(spy_b, model_id="model-b", root=tmp_path)

    enc_a.encode(["shared text", "other"])
    enc_b.encode(["shared text", "other"])

    assert spy_a.call_count == 1
    assert spy_b.call_count == 1  # would be 0 if b wrongly hit a's cache


def test_load_rejects_a_file_whose_embedded_model_id_does_not_match(tmp_path):
    """Belt-and-braces defense-in-depth: even if a cache file somehow ends
    up under the wrong model_id's directory (manual copy, future bug where
    two encoders share a root), a mismatch must raise loudly rather than
    silently serve the wrong vectors."""
    victim = CachedEncoder(SpyEncoder(), model_id="model-b", root=tmp_path)
    # Fabricate a file at exactly the path model-b would look up, but
    # written (via a real CachedEncoder) as if it belonged to model-a.
    target_path = victim._path("poisoned")
    tamperer = CachedEncoder(SpyEncoder(), model_id="model-a", root=tmp_path)
    tamperer._store(tamperer._path("poisoned"), _vec("poisoned"))
    shutil.copyfile(tamperer._path("poisoned"), target_path)

    with pytest.raises(ValueError, match="model_id"):
        victim.encode(["poisoned"])


# ---------------------------------------------------------------------------
# Property 2: return order matches input order, including under partial hit.
# ---------------------------------------------------------------------------

def test_order_matches_input_under_partial_hit(tmp_path):
    spy = SpyEncoder()
    enc = CachedEncoder(spy, model_id="m", root=tmp_path)
    enc.encode(["a", "b"])  # warm the cache for a and b

    spy2 = SpyEncoder()
    enc2 = CachedEncoder(spy2, model_id="m", root=tmp_path)
    out = enc2.encode(["b", "c", "a"])

    assert np.allclose(out[0], _vec("b"))
    assert np.allclose(out[1], _vec("c"))
    assert np.allclose(out[2], _vec("a"))
    # only "c" should have required a fresh compute
    assert spy2.calls == [["c"]]


# ---------------------------------------------------------------------------
# Property 3: duplicate texts inside one call are deduplicated.
# ---------------------------------------------------------------------------

def test_duplicate_texts_in_one_call_hit_inner_encode_once(tmp_path):
    spy = SpyEncoder()
    enc = CachedEncoder(spy, model_id="m", root=tmp_path)
    out = enc.encode(["dup", "other", "dup"])

    assert spy.call_count == 1
    assert sorted(spy.calls[0]) == ["dup", "other"]
    assert np.allclose(out[0], out[2])
    assert np.allclose(out[0], _vec("dup"))


# ---------------------------------------------------------------------------
# Property 4: inner.encode is not called at all on a full hit.
# ---------------------------------------------------------------------------

def test_full_hit_calls_inner_encode_zero_times(tmp_path):
    spy = SpyEncoder()
    enc = CachedEncoder(spy, model_id="m", root=tmp_path)
    texts = ["a", "b", "c"]
    enc.encode(texts)
    assert spy.call_count == 1

    out = enc.encode(texts)
    assert spy.call_count == 1  # unchanged: zero NEW calls
    assert np.allclose(out[0], _vec("a"))
    assert np.allclose(out[2], _vec("c"))


def test_full_hit_across_process_boundary_uses_only_disk(tmp_path):
    """Simulates the two-process scenario: a fresh CachedEncoder instance
    (as if in a new run_m1.py invocation) hitting a cache built earlier."""
    warm = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    warm.encode(["a", "b", "c"])

    spy2 = SpyEncoder()
    cold = CachedEncoder(spy2, model_id="m", root=tmp_path)
    out = cold.encode(["a", "b", "c"])
    assert spy2.call_count == 0
    assert np.allclose(out[1], _vec("b"))


# ---------------------------------------------------------------------------
# Property 5: atomic writes; corrupt/truncated files are detected & rebuilt.
# ---------------------------------------------------------------------------

def test_write_uses_a_temp_file_and_os_replace_leaving_no_tmp_behind(tmp_path):
    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    enc.encode(["a", "b"])
    leftover_tmp = list((tmp_path / "m").glob("*.tmp-*"))
    assert leftover_tmp == []
    real_files = list((tmp_path / "m").glob("*.npz"))
    assert len(real_files) == 2


def test_target_path_is_published_by_an_atomic_rename(tmp_path, monkeypatch):
    """Atomicity is a syscall-level contract, so assert it at that level.

    Leaving no .tmp behind is necessary but nowhere near sufficient: a
    `shutil.copyfile(tmp, path); unlink(tmp)` implementation also leaves no
    tmp behind, yet a concurrent reader can observe `path` half-written.
    Only a rename publishes the entry indivisibly, so pin the rename.
    """
    import amrag.index.cache as cache_mod

    renames: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        renames.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(cache_mod.os, "replace", spy)
    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    enc.encode(["a", "b"])

    assert len(renames) == 2, "each entry must be published by exactly one rename"
    for src, dst in renames:
        assert dst.endswith(".npz")
        assert ".tmp-" in src, "must rename FROM a temp file, not write in place"
        assert os.path.dirname(src) == os.path.dirname(dst), (
            "temp file must live in the destination directory, or the rename "
            "can cross a filesystem boundary and stop being atomic"
        )


def test_a_crash_mid_serialisation_leaves_no_file_at_the_target_path(tmp_path):
    """The SIGINT-mid-write case. A reader must never find a torn file."""
    import amrag.index.cache as cache_mod

    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    path = enc._path("a")

    def exploding_savez(fh, **kw):
        fh.write(b"PK\x03\x04 partial npz header")   # realistic torn write
        raise KeyboardInterrupt("simulated SIGINT mid-write")

    original = cache_mod.np.savez
    cache_mod.np.savez = exploding_savez
    try:
        with pytest.raises(KeyboardInterrupt):
            enc.encode(["a", "b"])
    finally:
        cache_mod.np.savez = original

    assert not path.exists(), "a torn file was published under the real name"
    assert list((tmp_path / "m").glob("*.tmp-*")) == [], "temp file not cleaned up"

    # and the entry is simply recomputed on the next run
    spy = SpyEncoder()
    out = CachedEncoder(spy, model_id="m", root=tmp_path).encode(["a"])
    assert np.allclose(out[0], _vec("a"))


def test_corrupt_cache_file_is_detected_and_rebuilt(tmp_path):
    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    enc.encode(["a", "b"])  # write both to disk

    path_a = enc._path("a")
    path_a.write_bytes(b"not a valid npz file, truncated garbage")

    spy2 = SpyEncoder()
    enc2 = CachedEncoder(spy2, model_id="m", root=tmp_path)
    out = enc2.encode(["a", "b"])

    # corrupted "a" was recomputed (not "b", which was intact)
    assert spy2.calls == [["a"]]
    assert np.allclose(out[0], _vec("a"))
    assert np.allclose(out[1], _vec("b"))

    # and the corrupt file was rebuilt: a third, cold encoder now hits both
    spy3 = SpyEncoder()
    enc3 = CachedEncoder(spy3, model_id="m", root=tmp_path)
    enc3.encode(["a", "b"])
    assert spy3.call_count == 0


def test_missing_file_is_an_ordinary_miss_not_an_error(tmp_path):
    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    out = enc.encode(["never seen before", "also new"])
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Property 6: vectors round-trip L2-normalised, stored as float32.
# ---------------------------------------------------------------------------

def test_vectors_round_trip_l2_normalised_and_as_float32(tmp_path):
    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    enc.encode(["a", "b"])

    cold = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    out = cold.encode(["a", "b"])

    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_stored_file_is_float32_not_float16(tmp_path):
    enc = CachedEncoder(SpyEncoder(), model_id="m", root=tmp_path)
    enc.encode(["a", "b"])
    with np.load(enc._path("a")) as data:
        assert data["vector"].dtype == np.float32


# ---------------------------------------------------------------------------
# Query-side write decision: "cache only what build() passes".
# ---------------------------------------------------------------------------

def test_singleton_calls_are_not_persisted_to_disk(tmp_path):
    """A single-text call (the shape DenseRetriever.retrieve uses for query
    encoding) must not write a cache file -- see report for justification.
    It must still work: return the right vector and not crash."""
    spy = SpyEncoder()
    enc = CachedEncoder(spy, model_id="m", root=tmp_path)
    out = enc.encode(["one-shot hyde text"])
    assert np.allclose(out[0], _vec("one-shot hyde text"))
    assert list((tmp_path / "m").glob("*.npz")) == []

    # a second identical singleton call recomputes rather than reading a
    # cache that was never written -- proves it, not just asserts the dir
    spy2_calls_before = spy.call_count
    enc.encode(["one-shot hyde text"])
    assert spy.call_count == spy2_calls_before + 1


def test_batch_calls_are_persisted_even_when_a_singleton_call_happened_first(tmp_path):
    spy = SpyEncoder()
    enc = CachedEncoder(spy, model_id="m", root=tmp_path)
    enc.encode(["query text"])  # singleton: not persisted
    enc.encode(["doc1", "doc2", "doc3"])  # batch: persisted

    assert list((tmp_path / "m").glob("*.npz")) != []
    spy_cold = SpyEncoder()
    cold = CachedEncoder(spy_cold, model_id="m", root=tmp_path)
    cold.encode(["doc1", "doc2", "doc3"])
    assert spy_cold.call_count == 0  # full hit on the batch


# ---------------------------------------------------------------------------
# Integration shape: drops into DenseRetriever.build unchanged.
# ---------------------------------------------------------------------------

def test_works_as_encoder_for_dense_retriever_build(tmp_path):
    docs = [Document("d1", "cat", {}), Document("d2", "photon", {})]
    spy = SpyEncoder()
    enc = CachedEncoder(spy, model_id="m", root=tmp_path)
    retriever = DenseRetriever.build(docs, enc)
    hits = retriever.retrieve("cat", k=1)
    assert hits[0].doc_id == "d1"

    # a second build over the same docs, fresh process-like encoder: full hit
    spy2 = SpyEncoder()
    enc2 = CachedEncoder(spy2, model_id="m", root=tmp_path)
    DenseRetriever.build(docs, enc2)
    assert spy2.call_count == 0
