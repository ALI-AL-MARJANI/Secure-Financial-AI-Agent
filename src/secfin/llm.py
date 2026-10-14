"""Thin wrapper around a local Ollama chat model. Everything that calls an LLM takes a `TextLLM`
so tests can pass a deterministic fake."""

from __future__ import annotations

from typing import Protocol


class TextLLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


class OllamaLLM:
    def __init__(
        self,
        model: str = "mistral",
        temperature: float = 0.0,
        seed: int | None = None,
        num_ctx: int = 8192,
        base_url: str | None = None,
    ):
        from langchain_ollama import ChatOllama

        kwargs = {"model": model, "temperature": temperature, "num_ctx": num_ctx}
        if seed is not None:
            kwargs["seed"] = seed
        if base_url:
            kwargs["base_url"] = base_url
        self.model = model
        self.chat = ChatOllama(**kwargs)

    def __call__(self, prompt: str) -> str:
        return str(self.chat.invoke(prompt).content).strip()
