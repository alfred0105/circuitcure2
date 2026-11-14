"""Small standard-library client for a loopback-only Ollama server."""

import base64
import json
import os
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LocalLLMError(RuntimeError):
    """A local Ollama request failed or returned unusable data."""


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 180,
                 num_predict: int = 1300, keep_alive: str = "10m", metadata_ttl_seconds: float = 30,
                 auto_start: bool = False):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama must use a local loopback HTTP URL")
        if type(num_predict) is not int or num_predict < 1:
            raise ValueError("num_predict must be a positive integer")
        self.backend = os.environ.get("CIRCUITCARE_LLM_BACKEND", "ollama")
        if self.backend == "mlx" and base_url == "http://127.0.0.1:11434":
            base_url = "http://127.0.0.1:8766"
        self.base_url = base_url.rstrip("/")
        self.auto_start = auto_start
        self.timeout_seconds = timeout_seconds
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.metadata_ttl_seconds = max(0.0, float(metadata_ttl_seconds))
        self._metadata_lock = threading.RLock()
        self._metadata_cache: dict[str, tuple[float, Any]] = {}

    def list_models(self) -> list[dict]:
        response = self._request("GET", "/api/tags")
        models = response.get("models")
        if not isinstance(models, list):
            raise LocalLLMError("Ollama /api/tags returned no models list")
        return models

    def show_model(self, model_name: str) -> dict:
        """Return Ollama /api/show metadata, including declared model capabilities."""
        return self._cached_metadata(f"show:{model_name}",
                                     lambda: self._request("POST", "/api/show", {"model": model_name}))

    def model_supports_vision(self, model_name: str) -> bool:
        metadata = self.show_model(model_name)
        capabilities = metadata.get("capabilities", [])
        return isinstance(capabilities, list) and "vision" in capabilities

    def model_info(self, model_name: str) -> dict:
        models = self._cached_metadata("tags", self.list_models)
        match = next((item for item in models if item.get("name") == model_name or item.get("model") == model_name), None)
        if match is None and ":" not in model_name:
            match = next((item for item in models if item.get("name", "").split(":", 1)[0] == model_name), None)
        if match is None:
            available = ", ".join(item.get("name", "") for item in models) or "none"
            raise LocalLLMError(f"Model {model_name!r} is not installed locally (available: {available})")
        return match

    def invalidate_model_metadata(self) -> None:
        """Clear cached model capabilities after the local Ollama service changes."""
        with self._metadata_lock:
            self._metadata_cache.clear()

    def unload_model(self, model_name: str) -> bool:
        """Unload an Ollama model after a bounded workflow; native MLX servers manage their own lifetime."""
        if self.backend != "ollama":
            return False
        self._request("POST", "/api/generate", {
            "model": model_name, "prompt": "", "stream": False, "keep_alive": 0,
        })
        return True

    def _cached_metadata(self, key: str, loader):
        now = time.monotonic()
        with self._metadata_lock:
            cached = self._metadata_cache.get(key)
            if cached and now - cached[0] < self.metadata_ttl_seconds:
                return cached[1]
        value = loader()
        with self._metadata_lock:
            self._metadata_cache[key] = (time.monotonic(), value)
        return value

    def chat_json(self, model_name: str, messages: list[dict], *,
                  output_schema: dict | None = None,
                  images: list[bytes | str] | None = None,
                  output_budget: int | None = None) -> dict:
        model_info = self.model_info(model_name)
        request_messages = [dict(message) for message in messages]
        if images:
            encoded_images = []
            for image in images:
                if isinstance(image, bytes):
                    encoded = base64.b64encode(image).decode("ascii")
                elif isinstance(image, str):
                    encoded = image
                    try:
                        base64.b64decode(encoded, validate=True)
                    except ValueError as exc:
                        raise ValueError("Ollama image input must be bytes or valid base64.") from exc
                else:
                    raise TypeError("Ollama image input must be bytes or base64 text.")
                if not encoded:
                    raise ValueError("Ollama image input cannot be empty.")
                encoded_images.append(encoded)
            image_message = next(
                (message for message in reversed(request_messages) if message.get("role") == "user"),
                None,
            )
            if image_message is None:
                image_message = {"role": "user", "content": "Inspect the attached image."}
                request_messages.append(image_message)
            image_message["images"] = encoded_images
        body = {
            "model": model_name,
            "messages": request_messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "format": output_schema or "json",
            "options": {"temperature": 0.1,
                        "num_predict": output_budget if type(output_budget) is int and output_budget > 0
                        else self.num_predict},
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
                reason = response.get("done_reason")
                detail = "; response may have been truncated at the output token limit" if reason == "length" else ""
                diagnostics = ", ".join(
                    f"{key}={response.get(key)}" for key in
                    ("done_reason", "eval_count", "prompt_eval_count") if response.get(key) is not None
                )
                if diagnostics:
                    detail += f"; Ollama generation diagnostics: {diagnostics}"
                raise LocalLLMError(
                    f"Ollama returned invalid JSON content: {exc}{detail}; response starts {content[:240]!r}"
                ) from exc
        if not isinstance(parsed, dict):
            # Vision models occasionally serialize the requested array value as
            # the top-level JSON array even when the schema wraps it in an
            # object, e.g. [{"detection_id":"1", ...}] instead of
            # {"class_reviews":[...]}. Recover only when the schema identifies
            # one unambiguous array field; downstream stage validation still
            # checks IDs, counts, and required rows.
            if isinstance(parsed, list) and isinstance(output_schema, dict):
                properties = output_schema.get("properties")
                required = output_schema.get("required", [])
                if isinstance(properties, dict):
                    array_keys = [key for key, spec in properties.items()
                                  if isinstance(spec, dict) and spec.get("type") == "array"]
                    constrained_keys = [key for key in array_keys
                                        if int(properties[key].get("minItems", 0) or 0) > 0]
                    target_keys = constrained_keys if len(constrained_keys) == 1 else array_keys
                    if len(target_keys) == 1 and target_keys[0] in required:
                        normalized = {target_keys[0]: parsed}
                        for key in required:
                            if key != target_keys[0] and key in properties \
                                    and isinstance(properties[key], dict) \
                                    and properties[key].get("type") == "array":
                                normalized[key] = []
                        parsed = normalized
            if not isinstance(parsed, dict):
                raise LocalLLMError(f"Ollama JSON content must match the requested object schema (received {type(parsed).__name__})")
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
            "load_duration_ns": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
            "eval_count": response.get("eval_count"),
            "eval_duration_ns": response.get("eval_duration"),
        }

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if self.backend == "mlx":
            from circuitcare.mlx_transport import request_mlx
            try:
                return request_mlx(self._http_request, method, path, payload)
            except (ValueError, OSError) as exc:
                raise LocalLLMError(f"Native MLX request failed: {exc}") from exc
        return self._http_request(method, path, payload)

    def _http_request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            from circuitcare.ollama_runtime import ollama_request
            with ollama_request(self.base_url, auto_start=self.auto_start):
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
    # Models may hit the output limit before emitting the closing fence. Drop the
    # opening marker anyway so a complete JSON object can still be recovered.
    match = re.match(r"^```(?:json)?\s*([\s\S]*)$", stripped, flags=re.IGNORECASE)
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
