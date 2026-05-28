"""OpenAI-compatible LLM client for vLLM and Ollama backends."""

from dataclasses import dataclass, field

import httpx
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
    timeout: float = 600.0  # 10 minutes — Modal cold starts can be slow
    max_retries: int = 3
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self):
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout, connect=30.0),
            max_retries=self.max_retries,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion. Optionally constrain with EBNF grammar.

        Args:
            messages: OpenAI-style message list [{role, content}, ...]
            grammar: EBNF string for XGrammar constrained decoding (vLLM only)
            max_tokens: Override default max_tokens for this call
            temperature: Override default temperature for this call

        Returns:
            The assistant's response text.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": True,
        }
        if grammar:
            kwargs["extra_body"] = {"guided_grammar": grammar}

        # Stream to avoid Modal proxy timeouts on long generations
        chunks = []
        for chunk in self._client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)

    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Generate a text completion from a prompt prefix.

        Uses the /v1/completions endpoint (not chat). Designed for base
        models that do next-token prediction rather than instruction following.

        Args:
            prompt: The text prefix to complete
            grammar: EBNF string for XGrammar constrained decoding
            max_tokens: Override default max_tokens for this call
            temperature: Override default temperature for this call
            stop: Stop sequences to end generation early

        Returns:
            The generated completion text.
        """
        kwargs: dict = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": True,
        }
        if grammar:
            kwargs["extra_body"] = {"guided_grammar": grammar}
        if stop:
            kwargs["stop"] = stop

        chunks = []
        for chunk in self._client.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].text:
                chunks.append(chunk.choices[0].text)
        return "".join(chunks)
