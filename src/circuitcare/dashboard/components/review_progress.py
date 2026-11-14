"""Render a stable browser progress surface backed by a job-scoped SSE stream."""
import hashlib
from pathlib import Path
from circuitcare.vision.review_progress_stream import progress_server, public_progress

_COMPONENT = None


def render_review_progress(st, *, owner, job_id, job, on_completed):
    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = st.components.v2.component(
            name='circuitcare_review_progress',
            html='<div data-review-progress><strong>AI 검수</strong><span data-stage></span><progress max="1" value="0"></progress><span data-elapsed></span><span data-detail></span></div>',
            css='\nbody{margin:0}[data-review-progress]{box-sizing:border-box;height:36px;font:13px var(--st-font);color:var(--st-text-color);display:flex;align-items:center;gap:9px;padding:0;overflow:hidden;white-space:nowrap}\n[data-review-progress]>strong,[data-stage],[data-elapsed]{flex:0 0 auto}[data-stage]{max-width:34%;overflow:hidden;text-overflow:ellipsis}progress{display:block;box-sizing:border-box;flex:0 0 76px;width:76px;height:7px;accent-color:#68b3ff}[data-detail]{flex:1;min-width:0;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
            js=(Path(__file__).with_name('assets') / 'review_progress.js').read_text())
    server = progress_server()
    token = server.bind(owner, job_id)
    key = 'cc_review_stream_' + hashlib.sha256(f'{owner}|{job_id}'.encode()).hexdigest()[:20]
    def callback(result_override=None):
        result = result_override if result_override is not None else st.session_state.get(key)
        event = result.get('completed') if isinstance(result, dict) else getattr(result, 'completed', None)
        if isinstance(event, dict) and event.get('job_id') == job_id and event.get('token') == token:
            handled_key = key + '_completed'
            if not st.session_state.get(handled_key):
                st.session_state[handled_key] = True
                on_completed()
    result = _COMPONENT(data={'job_id':job_id,'token':token,
                           'url':f'http://127.0.0.1:{server.http.server_port}/events/{token}',
                           'initial':public_progress(job)},
                      key=key, default={'completed':None}, on_completed_change=callback,
                      width='stretch', height=38)
    callback(result)
    return result
