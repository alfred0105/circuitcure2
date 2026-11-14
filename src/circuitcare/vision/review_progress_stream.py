"""Loopback SSE progress transport; never exposes worker state or PCB data."""
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time
from functools import lru_cache
from urllib.parse import urlsplit

from circuitcare.vision.ai_review_jobs import get_ai_review_job, JOB_CHANGED

_ORIGINS = {'http://localhost:8765', 'http://127.0.0.1:8765'}


def public_progress(job):
    if job is None:
        return {'status': 'UNAVAILABLE', 'progress': {'message': '작업 상태를 찾을 수 없습니다.'}}
    return {key: job.get(key) for key in ('job_id', 'status', 'progress', 'elapsed_ms', 'error')}


class ProgressServer:
    def __init__(self):
        self.lock = threading.RLock()
        self.tokens = OrderedDict()
        registry = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args): pass
            def do_GET(self):
                origin = self.headers.get('Origin')
                if origin is not None and origin not in _ORIGINS:
                    self.send_error(403)
                    return
                path = urlsplit(self.path).path
                with registry.lock:
                    binding = registry.tokens.get(path.removeprefix('/events/')) if path.startswith('/events/') else None
                if binding is None or time.monotonic() - binding[2] > 86400:
                    self.send_error(404)
                    return
                owner, job_id, _created = binding
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                if origin:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')
                self.end_headers()
                previous = None
                try:
                    while True:
                        with JOB_CHANGED:
                            version = JOB_CHANGED.version
                        job = public_progress(get_ai_review_job(owner=owner, job_id=job_id))
                        signature = json.dumps({k: v for k, v in job.items() if k != 'elapsed_ms'}, sort_keys=True)
                        if signature != previous:
                            self.wfile.write(('data: ' + json.dumps(job, ensure_ascii=False) + '\n\n').encode())
                            self.wfile.flush()
                            previous = signature
                        if job.get('status') not in {'PENDING', 'RUNNING'}:
                            break
                        with JOB_CHANGED:
                            if JOB_CHANGED.version == version:
                                JOB_CHANGED.wait(timeout=15)
                        self.wfile.write(b': keepalive\n\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True,
                         name='pcb-review-progress-stream').start()

    def bind(self, owner, job_id):
        with self.lock:
            now = time.monotonic()
            for token, row in list(self.tokens.items()):
                if now - row[2] > 86400:
                    self.tokens.pop(token)
            for token, row in self.tokens.items():
                if row[:2] == (owner, job_id):
                    return token
            token = secrets.token_urlsafe(32)
            self.tokens[token] = (owner, job_id, now)
            while len(self.tokens) > 128:
                self.tokens.popitem(last=False)
            return token


@lru_cache(maxsize=1)
def progress_server():
    return ProgressServer()
