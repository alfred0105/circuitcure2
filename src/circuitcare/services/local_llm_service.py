"""Shared local Ollama availability lookup."""

from __future__ import annotations

from circuitcare.local_llm import LocalLLMError, OllamaClient


def list_local_models():
    try:
        client = OllamaClient(auto_start=True, timeout_seconds=180)
        return client, client.list_models()
    except (LocalLLMError, ValueError):
        return None, []


