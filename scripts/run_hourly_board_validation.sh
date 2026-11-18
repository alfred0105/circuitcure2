#!/bin/zsh
set -euo pipefail

ROOT="${CIRCUITCARE_REPO_ROOT:-$HOME/Documents/ChatGPT/circuitcure2}"
DATA_ROOT="$HOME/.local/share/circuitcare-ai/external_datasets/pcb-dslr"
MANIFEST="$DATA_ROOT/holdout-50.json"
RUN_ROOT="$DATA_ROOT/hourly-runs"
LOCK="$DATA_ROOT/hourly-validation.lock"
PYTHON_BIN="${CIRCUITCARE_VALIDATION_PYTHON:-$HOME/Documents/ChatGPT/대회/.venv-component-train/bin/python}"
DASHBOARD="$ROOT/data/validation/dashboard-live.json"
BASELINE="$DATA_ROOT/baseline/dashboard-live.json"

if ! mkdir "$LOCK" 2>/dev/null; then
  print -r -- "$(date -u +%FT%TZ) already running; skip overlapping hourly run"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CIRCUITCARE_VALIDATION_HOLDOUT_MANIFEST="$MANIFEST"

if [[ ! -f "$MANIFEST" ]]; then
  print -r -- "Missing board-level holdout manifest: $MANIFEST" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  print -r -- "Python environment unavailable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -f "$BASELINE" ]]; then
  mkdir -p "$(dirname "$BASELINE")"
  CIRCUITCARE_USE_TRAINED_CHECKPOINTS=0 "$PYTHON_BIN" -m circuitcare.benchmark.pcb_validation_cli run \
    "$MANIFEST" "$DATA_ROOT/baseline/runs" --real-yolo --skip-qwen --dashboard \
    --dashboard-path "$BASELINE"
fi

RUN_DIR="$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
if "$PYTHON_BIN" -m circuitcare.benchmark.pcb_validation_cli run \
    "$MANIFEST" "$RUN_DIR" --real-yolo --skip-qwen --dashboard \
    --dashboard-path "$DASHBOARD" --baseline-dashboard "$BASELINE"; then
  "$PYTHON_BIN" - "$DASHBOARD" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8"))
d["run_state"] = "COMPLETE"
d["last_updated"] = datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
else
  "$PYTHON_BIN" - "$DASHBOARD" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1])
try: d = json.loads(p.read_text(encoding="utf-8"))
except Exception: d = {}
d.update(run_state="FAILED", error="홀드아웃 평가 명령 실패", last_updated=datetime.now(timezone.utc).isoformat())
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  exit 1
fi
