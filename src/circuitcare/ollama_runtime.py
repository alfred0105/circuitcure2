"""On-demand ownership of the selected local Ollama or native MLX server."""
from contextlib import contextmanager
import atexit
import os
import shutil
import sys
from types import ModuleType
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

# Streamlit reloads application modules during development. Keep ownership and
# active leases in a process-local module without a source file so a reload
# cannot create an independent idle timer that stops another request.
_BACKEND = os.environ.get("CIRCUITCARE_LLM_BACKEND", "ollama")
_PORT = 8766 if _BACKEND == "mlx" else 11434
_HEALTH = f"http://127.0.0.1:{_PORT}" + ("/v1/models" if _BACKEND == "mlx" else "/api/version")
_STATE = sys.modules.setdefault(f"_circuitcare_model_process_state_{_BACKEND}", ModuleType(f"_circuitcare_model_process_state_{_BACKEND}"))
if not hasattr(_STATE, "lock"):
    _STATE.lock = threading.RLock()
    _STATE.process = None
    _STATE.active = 0
    _STATE.timer = None
    _STATE.prewarming = False
elif not hasattr(_STATE, "prewarming"):
    # This module state survives Streamlit code reloads.
    _STATE.prewarming = False
_IDLE_SECONDS = 120


def _ready():
    try:
        with urlopen(_HEALTH, timeout=0.5) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _stop_idle():
    with _STATE.lock:
        _STATE.timer = None
        if _STATE.active or _STATE.process is None:
            return
        process, _STATE.process = _STATE.process, None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _schedule_idle():
    if _STATE.process is not None and not _STATE.active:
        if _STATE.timer is not None:
            _STATE.timer.cancel()
        _STATE.timer = threading.Timer(_IDLE_SECONDS, _stop_idle)
        _STATE.timer.daemon = True
        _STATE.timer.start()


def _start():
    if _ready():
        return  # An independently started server is never ours to terminate.
    binary = shutil.which(os.environ.get("CIRCUITCARE_MLX_PYTHON", "python") if _BACKEND == "mlx" else "ollama")
    if not binary:
        raise OSError("선택한 로컬 AI 서버의 실행 파일을 찾을 수 없습니다.")
    if _STATE.process is None or _STATE.process.poll() is not None:
        command = ([binary, "-m", "mlx_vlm.server", "--host", "127.0.0.1", "--port", "8766",
                    "--model", os.environ.get("CIRCUITCARE_MLX_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit"),
                    "--max-num-seqs", "1", "--vision-cache-size", "16"] if _BACKEND == "mlx" else [binary, "serve"])
        _STATE.process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + (90 if _BACKEND == "mlx" else 15)
    while time.monotonic() < deadline:
        if _ready():
            return
        if _STATE.process.poll() is not None:
            raise OSError("로컬 AI 서버가 시작 중 종료되었습니다.")
        time.sleep(0.1)
    raise OSError("로컬 AI 서버 시작 시간이 초과됐습니다.")


def prewarm_model_server():
    """Load the configured local model in the background after photo intake."""
    if _ready():
        return False
    with _STATE.lock:
        if _STATE.prewarming:
            return False
        _STATE.prewarming = True

    def warm():
        try:
            with _STATE.lock:
                _start()
                _schedule_idle()
        except OSError:
            # The real AI action reports a detailed startup error if the user
            # later requests inference; intake and YOLO remain available.
            pass
        finally:
            with _STATE.lock:
                _STATE.prewarming = False

    threading.Thread(target=warm, name="circuitcare-model-prewarm", daemon=True).start()
    return True


@contextmanager
def ollama_request(base_url, *, auto_start=False):
    managed_endpoint = base_url.rstrip("/") in {f"http://127.0.0.1:{_PORT}", f"http://localhost:{_PORT}"}
    if not managed_endpoint:
        yield
        return
    with _STATE.lock:
        _STATE.active += 1
        if _STATE.timer is not None:
            _STATE.timer.cancel()
            _STATE.timer = None
    try:
        if auto_start:
            with _STATE.lock:
                _start()
        yield
    finally:
        with _STATE.lock:
            _STATE.active -= 1
            _schedule_idle()


atexit.register(_stop_idle)
