from types import SimpleNamespace
import pytest
from circuitcare import ollama_runtime as runtime


@pytest.fixture
def managed(monkeypatch):
    monkeypatch.setattr(runtime._STATE, "process", None)
    monkeypatch.setattr(runtime._STATE, "active", 0)
    monkeypatch.setattr(runtime._STATE, "timer", None)
    timers = []
    class Timer:
        def __init__(self, delay, fn):
            self.delay, self.fn, self.cancelled = delay, fn, False
            timers.append(self)
        def start(self): pass
        def cancel(self): self.cancelled = True
    monkeypatch.setattr(runtime.threading, "Timer", Timer)
    return timers


def test_external_server_is_never_owned_or_stopped(monkeypatch, managed):
    monkeypatch.setattr(runtime, "_ready", lambda: True)
    with runtime.ollama_request("http://127.0.0.1:11434", auto_start=True):
        assert runtime._STATE.active == 1
    assert runtime._STATE.process is None and managed == []


def test_managed_server_survives_active_requests_then_stops_when_idle(monkeypatch, managed):
    calls = []
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"),
                              wait=lambda **_kw: calls.append("wait"))
    ready = iter([False, True])
    monkeypatch.setattr(runtime, "_ready", lambda: next(ready))
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/bin/ollama")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: process)
    with runtime.ollama_request("http://127.0.0.1:11434", auto_start=True):
        runtime._stop_idle()
        assert calls == []
    assert managed[-1].delay == 120
    with runtime.ollama_request("http://127.0.0.1:11434"):
        assert managed[0].cancelled
        runtime._stop_idle()
        assert calls == []
    managed[-1].fn()
    assert calls == ["terminate", "wait"]


def test_failed_start_releases_request_lease(monkeypatch, managed):
    monkeypatch.setattr(runtime, "_ready", lambda: False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    with pytest.raises(OSError):
        with runtime.ollama_request("http://127.0.0.1:11434", auto_start=True):
            pass
    assert runtime._STATE.active == 0


def test_custom_endpoint_does_not_start_default_server(monkeypatch, managed):
    monkeypatch.setattr(runtime, "_start", lambda: pytest.fail("unexpected startup"))
    with runtime.ollama_request("http://localhost:12345", auto_start=True):
        assert runtime._STATE.active == 0


def test_module_reload_preserves_active_request_ownership(managed):
    import importlib
    state = runtime._STATE
    with runtime.ollama_request("http://127.0.0.1:11434"):
        importlib.reload(runtime)
        assert runtime._STATE is state
        assert runtime._STATE.active == 1
    assert runtime._STATE.active == 0
