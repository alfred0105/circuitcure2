"""Shared local Ollama availability lookup."""

from __future__ import annotations

from circuitcare.local_llm import LocalLLMError, OllamaClient


def list_local_models():
    try:
        client = OllamaClient(timeout_seconds=180)
        return client, client.list_models()
    except (LocalLLMError, ValueError):
        return None, []


