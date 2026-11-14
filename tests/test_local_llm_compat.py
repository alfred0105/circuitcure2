from __future__ import annotations

import pytest

from circuitcare.local_llm import LocalLLMError, OllamaClient


def _response(content: str = '{"ok":true}') -> dict:
    return {"model": "test:latest", "message": {"content": content}, "done": True}


def test_chat_json_falls_back_only_for_unsupported_structured_output(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest", "digest": "abc"})
    calls = []

    def request(_method, _path, payload):
        calls.append(payload.copy())
        if len(calls) == 1:
            raise LocalLLMError('Ollama request failed with HTTP 501: {"error":"structured output is unavailable"}')
        return _response()

    monkeypatch.setattr(client, "_request", request)
    result = client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])
    assert result["content"] == {"ok": True}
    assert result["json_mode"] is False
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "structured_output_unavailable"
    assert calls[0]["format"] == "json" and "format" not in calls[1]
    assert calls[1]["think"] is False


@pytest.mark.parametrize("error", [
    'Ollama request failed with HTTP 501: {"error":"route unavailable"}',
    'Could not reach local Ollama at http://127.0.0.1:11434: connection refused',
    'Ollama request failed with HTTP 404: model not found',
])
def test_chat_json_does_not_fallback_for_other_errors(monkeypatch, error):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = []

    def request(*_args):
        calls.append(1)
        raise LocalLLMError(error)

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(LocalLLMError):
        client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])
    assert len(calls) == 1


def test_chat_json_rejects_invalid_fallback_json(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = iter([LocalLLMError("structured output is unavailable"), _response("not JSON")])
    monkeypatch.setattr(client, "_request", lambda *_args: (_raise(next(calls))))
    with pytest.raises(LocalLLMError, match="invalid JSON"):
        client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])


@pytest.mark.parametrize("content", [
    '```json\n{"ok":true}\n```',
    '```json\n{"ok":true}',
    'Result:\n```json\n{"ok":true}\n```',
])
def test_chat_json_recovers_complete_json_from_fenced_model_output(monkeypatch, content):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    monkeypatch.setattr(client, "_request", lambda *_args: _response(content))
    assert client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])["content"] == {"ok": True}


def test_chat_json_uses_configured_output_token_budget_and_reports_truncation(monkeypatch):
    client = OllamaClient(num_predict=4096)
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = []

    def request(_method, _path, payload):
        calls.append(payload)
        return {**_response('```json\n{"incomplete":'), "done_reason": "length"}

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(LocalLLMError, match="truncated at the output token limit"):
        client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])
    assert calls[0]["options"]["num_predict"] == 4096


def test_chat_json_keeps_model_warm_and_exposes_ollama_performance_metrics(monkeypatch):
    client = OllamaClient(num_predict=8192, keep_alive="10m")
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = []
    monkeypatch.setattr(client, "_request", lambda _method, _path, payload: (
        calls.append(payload) or {**_response(), "total_duration": 2_000_000,
        "load_duration": 250_000, "prompt_eval_count": 12,
        "prompt_eval_duration": 500_000, "eval_count": 5, "eval_duration": 750_000}
    ))
    result = client.chat_json("test:latest", [{"role": "user", "content": "review"}], output_budget=700)
    assert calls[0]["keep_alive"] == "10m"
    assert calls[0]["options"]["num_predict"] == 700
    assert result["total_duration_ns"] == 2_000_000
    assert result["prompt_eval_count"] == 12
    assert result["eval_count"] == 5


def test_unload_model_requests_immediate_ollama_unload(monkeypatch):
    client = OllamaClient()
    calls = []
    monkeypatch.setattr(client, "_request", lambda method, path, payload: (
        calls.append((method, path, payload)) or {"done": True}
    ))
    assert client.unload_model("qwen3.5:9b") is True
    assert calls == [("POST", "/api/generate", {
        "model": "qwen3.5:9b", "prompt": "", "stream": False, "keep_alive": 0,
    })]


def test_native_mlx_model_lifetime_is_not_manually_unloaded(monkeypatch):
    client = OllamaClient()
    client.backend = "mlx"
    monkeypatch.setattr(client, "_request", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    assert client.unload_model("qwen3.5:9b") is False


def test_model_metadata_is_cached_for_the_configured_ttl(monkeypatch):
    client = OllamaClient(metadata_ttl_seconds=30)
    model_calls = []
    show_calls = []
    monkeypatch.setattr(client, "list_models", lambda: (
        model_calls.append(1) or [{"name": "test:latest", "digest": "same"}]
    ))
    monkeypatch.setattr(client, "_request", lambda *_args: (
        show_calls.append(1) or {"capabilities": ["vision"]}
    ))
    assert client.model_info("test:latest")["digest"] == "same"
    assert client.model_info("test:latest")["digest"] == "same"
    assert client.model_supports_vision("test:latest") is True
    assert client.model_supports_vision("test:latest") is True
    assert len(model_calls) == 1
    assert len(show_calls) == 1


def _raise(value):
    if isinstance(value, Exception):
        raise value
    return value
