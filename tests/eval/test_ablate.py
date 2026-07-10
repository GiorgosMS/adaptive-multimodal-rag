from amrag.eval.ablate import ABLATION_LADDER, Config, to_markdown
import pytest


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
    with pytest.raises(ValueError):
        Config(name="x", sparse=False, dense=False, rerank=False, hyde=False)
