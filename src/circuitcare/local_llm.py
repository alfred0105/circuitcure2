"""Small standard-library client for a loopback-only Ollama server."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LocalLLMError(RuntimeError):
    """A local Ollama request failed or returned unusable data."""


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 180):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama must use a local loopback HTTP URL")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[dict]:
        response = self._request("GET", "/api/tags")
        models = response.get("models")
        if not isinstance(models, list):
            raise LocalLLMError("Ollama /api/tags returned no models list")
        return models

    def model_info(self, model_name: str) -> dict:
        models = self.list_models()
        match = next((item for item in models if item.get("name") == model_name or item.get("model") == model_name), None)
        if match is None and ":" not in model_name:
            match = next((item for item in models if item.get("name", "").split(":", 1)[0] == model_name), None)
        if match is None:
            available = ", ".join(item.get("name", "") for item in models) or "none"
            raise LocalLLMError(f"Model {model_name!r} is not installed locally (available: {available})")
        return match

    def chat_json(self, model_name: str, messages: list[dict[str, str]]) -> dict:
        model_info = self.model_info(model_name)
        body = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 1300},
        }
        json_mode = True
        fallback_reason = None
        try:
            response = self._request("POST", "/api/chat", body)
        except LocalLLMError as exc:
            if "structured output is unavailable" not in str(exc).casefold():
                raise
            # Some locally installed model runners do not implement Ollama's JSON mode.
            # The system prompt still requires JSON, which is parsed and validated below.
            body.pop("format", None)
            json_mode = False
            fallback_reason = "structured_output_unavailable"
            response = self._request("POST", "/api/chat", body)
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LocalLLMError("Ollama chat response did not include message.content")
        content = _strip_json_fence(content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            candidate = _extract_json_object(content)
            try:
                parsed = json.loads(candidate) if candidate is not None else None
            except json.JSONDecodeError:
                parsed = None
            if parsed is None:
                raise LocalLLMError(f"Ollama returned invalid JSON content: {exc}; response starts {content[:240]!r}") from exc
        if not isinstance(parsed, dict):
            raise LocalLLMError("Ollama JSON content must be an object")
        return {
            "content": parsed,
            "model": response.get("model", model_name),
            "model_info": {
                "name": model_info.get("name", model_name),
                "size": model_info.get("size"),
                "digest": model_info.get("digest"),
                "details": model_info.get("details", {}),
            },
            "json_mode": json_mode,
            "fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "done": response.get("done"),
            "done_reason": response.get("done_reason"),
            "total_duration_ns": response.get("total_duration"),
            "eval_count": response.get("eval_count"),
        }

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LocalLLMError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LocalLLMError(f"Could not reach local Ollama at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LocalLLMError(f"Ollama returned invalid response JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise LocalLLMError("Ollama response must be a JSON object")
        return result


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_json_object(content: str) -> str | None:
    start = content.find("{")
    if start < 0:
        return None
    try:
        _, end = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError:
        return None
    return content[start:start + end]
