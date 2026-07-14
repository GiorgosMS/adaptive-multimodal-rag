import threading
import time

import pytest

from amrag.retrieve.hyde import HYDE_PROMPT, hyde_transform, hyde_transform_batch


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


class PerQueryLLM:
    """Returns a distinct reply per query so a misordered result is visible.

    Thread-safe: hyde_transform_batch calls .complete concurrently, so a list
    append (which is atomic in CPython but not a guarantee to lean on) is
    guarded, and a lock records the true peak concurrency for the cap test.
    """
    def __init__(self, delay=0.0):
        self.delay = delay
        self._lock = threading.Lock()
        self.calls = 0
        self.concurrent = 0
        self.peak = 0

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.calls += 1
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
        try:
            if self.delay:
                time.sleep(self.delay)
            # The query text is the last line of HYDE_PROMPT.format(query=...).
            q = prompt.rsplit("Question: ", 1)[1].split("\n", 1)[0]
            return f"ANSWER_FOR::{q}"
        finally:
            with self._lock:
                self.concurrent -= 1


def test_batch_preserves_input_order():
    # The correctness-critical property: result[i] must be the expansion of
    # queries[i]. A misalignment here silently corrupts every downstream
    # retrieval metric, so it is the first thing pinned.
    llm = PerQueryLLM()
    queries = [f"q{i}" for i in range(50)]
    out = hyde_transform_batch(queries, llm, max_workers=8)
    assert out == [f"q{i}\n\nANSWER_FOR::q{i}" for i in range(50)]


def test_batch_matches_the_sequential_transform_exactly():
    queries = ["how are nets trained?", "what is the baseline?"]
    batched = hyde_transform_batch(queries, FakeLLM(), max_workers=4)
    sequential = [hyde_transform(q, FakeLLM()) for q in queries]
    assert batched == sequential


def test_batch_calls_the_llm_once_per_query():
    llm = PerQueryLLM()
    hyde_transform_batch([f"q{i}" for i in range(12)], llm, max_workers=4)
    assert llm.calls == 12


def test_batch_never_exceeds_max_workers_concurrent_calls():
    # Safety: with 1451 queries we must not open 1451 concurrent connections.
    # A small delay guarantees overlap so peak concurrency is meaningful.
    llm = PerQueryLLM(delay=0.02)
    hyde_transform_batch([f"q{i}" for i in range(40)], llm, max_workers=5)
    assert llm.peak <= 5, f"opened {llm.peak} concurrent calls, cap was 5"
    assert llm.peak >= 2, "no observed concurrency; the pool may be serialising"


def test_batch_empty_reply_falls_back_to_bare_query_per_element():
    out = hyde_transform_batch(["a", "b"], FakeLLM(reply="   "), max_workers=2)
    assert out == ["a", "b"]


def test_batch_of_no_queries_returns_empty():
    assert hyde_transform_batch([], FakeLLM(), max_workers=4) == []
