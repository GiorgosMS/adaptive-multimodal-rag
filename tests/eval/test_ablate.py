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


def test_markdown_missing_key_raises_valueerror():
    """A row missing a key derived from rows[0] raises ValueError with row index and missing keys."""
    rows = [{"config": "naive", "recall@10": 0.5}, {"config": "+hybrid"}]
    with pytest.raises(ValueError) as exc_info:
        to_markdown(rows)
    msg = str(exc_info.value)
    assert "row 1" in msg
    assert "recall@10" in msg
    assert "missing" in msg.lower()


def test_markdown_extra_key_raises_valueerror():
    """A row with an extra key not in rows[0] raises ValueError."""
    rows = [{"config": "naive", "recall@10": 0.5}, {"config": "+hybrid", "recall@10": 0.6, "extra": "value"}]
    with pytest.raises(ValueError) as exc_info:
        to_markdown(rows)
    msg = str(exc_info.value)
    assert "row 1" in msg
    assert "extra" in msg


def test_markdown_float_formatting():
    """Floats are formatted with :.4f, not as integers."""
    rows = [{"config": "naive", "recall@10": 0.5}]
    md = to_markdown(rows)
    assert "0.5000" in md
    assert "0.5" in md or "0.5000" in md


def test_markdown_int_renders_as_int_not_float():
    """An int value (e.g. {"missing_predictions": 3}) renders as 3, not 3.0000."""
    rows = [{"missing_predictions": 3}]
    md = to_markdown(rows)
    assert "3" in md
    assert "3.0000" not in md
    # Verify it's exactly "3" and not "3.0000" by checking the table contains " 3 |"
    assert "| 3 |" in md
