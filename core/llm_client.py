"""OpenAI-compatible LLM client for vLLM and Ollama backends."""

from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class LLMClient:
    """Thin wrapper around an OpenAI-compatible endpoint.

    Works with:
    - Ollama (localhost:11434/v1) for dev
    - vLLM local (localhost:8000/v1)
    - vLLM on Modal (remote URL)
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5-coder:7b"
    api_key: str = "ollama"  # Ollama ignores this but the SDK requires it
    max_tokens: int = 4096
    temperature: float = 0.0
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self):
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        grammar: str | None = None,
    ) -> str:
        """Generate a completion. Optionally constrain with EBNF grammar.

        Args:
            messages: OpenAI-style message list [{role, content}, ...]
            grammar: EBNF string for XGrammar constrained decoding (vLLM only)

        Returns:
            The assistant's response text.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if grammar:
            kwargs["extra_body"] = {"guided_grammar": grammar}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
