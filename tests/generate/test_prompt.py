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
