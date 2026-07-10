import pytest
from amrag.generate.llm import DeepSeekLLM
from amrag.retrieve.hyde import hyde_transform

class FakeChoice:
    def __init__(self, content): self.message = type("M", (), {"content": content})()

class FakeClient:
    def __init__(self): self.kwargs = None; self.chat = type("C", (), {"completions": self})()
    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("R", (), {"choices": [FakeChoice("hi")]})()

class NoneContentClient:
    """An OpenAI-compatible client whose response content is None (e.g. filtered)."""
    def __init__(self): self.chat = type("C", (), {"completions": self})()
    def create(self, **kwargs):
        return type("R", (), {"choices": [FakeChoice(None)]})()

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

def test_none_content_is_coerced_to_empty_string_not_none():
    out = DeepSeekLLM(client=NoneContentClient()).complete("hello")
    assert out == ""
    assert out is not None

def test_hyde_transform_composes_safely_with_a_none_content_response():
    assert hyde_transform("q", DeepSeekLLM(client=NoneContentClient())) == "q"
