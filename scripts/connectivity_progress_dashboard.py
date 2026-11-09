#!/usr/bin/env python3
"""Local live dashboard for connectivity experiments (stdlib only)."""
from __future__ import annotations

import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "data/benchmarks/phase22_connectivity/recommendation_progress.json"
PORT = 8766

PAGE = r"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CircuitCare 학습 진행률</title><style>
body{font:16px system-ui,sans-serif;background:#10151c;color:#e8edf2;margin:0;padding:32px}main{max-width:850px;margin:auto}
.card{background:#1b2530;border:1px solid #304050;border-radius:14px;padding:22px;margin:16px 0}h1{font-size:25px;margin:0}
.muted{color:#9aabba}.bar{height:18px;background:#0b1015;border-radius:20px;overflow:hidden;margin:18px 0 8px}
#fill{height:100%;width:0;background:linear-gradient(90deg,#36b6a0,#8ad86c);transition:width .5s}
.big{font-size:32px;font-weight:700}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.value{font-size:22px;font-weight:650}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style><main><h1>CircuitCare Connectivity 학습 상태</h1><p class="muted">2초마다 갱신 · 로컬 PC에서만 표시</p>
<section class="card"><div class="big" id="pct">대기 중</div><div class="bar"><div id="fill"></div></div><div id="phase">실험이 시작되면 진행 단계가 표시됩니다.</div><p id="detail" class="muted"></p></section>
<section class="card grid"><div><div class="muted">실험 단계</div><div id="step" class="value">—</div></div><div><div class="muted">GPU</div><div id="gpu" class="value">확인 중</div></div><div><div class="muted">GPU 메모리</div><div id="mem" class="value">—</div></div></section>
<section class="card"><div class="muted">최근 업데이트</div><div id="time">—</div><p id="status" class="muted"></p></section></main>
<script>async function tick(){try{let r=await fetch('/api/progress?t='+Date.now());let d=await r.json();
document.querySelector('#pct').textContent=d.overall_percent==null?'시작 대기':`${d.overall_percent.toFixed(1)}%`;
document.querySelector('#fill').style.width=`${d.overall_percent||0}%`;
document.querySelector('#phase').textContent=d.phase||'대기 중';document.querySelector('#detail').textContent=d.detail||'';
document.querySelector('#step').textContent=d.phase_number?`${d.phase_number} / ${d.phase_count}`:'—';
document.querySelector('#gpu').textContent=d.gpu?.name||'미검출';document.querySelector('#mem').textContent=d.gpu?.memory||'—';
document.querySelector('#time').textContent=d.updated_at||'—';document.querySelector('#status').textContent=d.status||'';
}catch(e){document.querySelector('#status').textContent='상태 파일을 기다리는 중';}}tick();setInterval(tick,2000);</script></html>"""


def gpu_info():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=2)
        name, util, used, total = out.strip().splitlines()[0].split(", ")
        return {"name": name, "utilization": f"{util}%", "memory": f"{used} / {total} MiB"}
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/progress"):
            try:
                data = json.loads(STATUS.read_text(encoding="utf-8"))
            except Exception:
                data = {"status": "대기 중", "overall_percent": 0, "phase": "학습 프로세스 시작 대기"}
            data["gpu"] = gpu_info()
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        body = PAGE.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print(f"Dashboard: http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
