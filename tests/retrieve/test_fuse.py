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
