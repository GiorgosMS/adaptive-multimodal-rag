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
