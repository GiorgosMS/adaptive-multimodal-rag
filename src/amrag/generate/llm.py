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
        content = r.choices[0].message.content
        return content if content is not None else ""
