import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from circuitcare.vision import review_progress_stream as stream
from circuitcare.vision.ai_review_jobs import JOB_CHANGED, _notify_progress


def read_event(response):
    while True:
        line = response.readline().decode().strip()
        if line.startswith('data: '):
            return json.loads(line[6:])
        if not line:
            continue


def test_stream_sends_only_progress_changes_and_terminal_event(monkeypatch):
    state = {'job_id':'j1','status':'RUNNING','elapsed_ms':1,
             'progress':{'stage':'OCR_BOARD','message':'OCR'}, 'state':{'private':'secret'}}
    monkeypatch.setattr(stream, 'get_ai_review_job', lambda **_kw: dict(state))
    server = stream.ProgressServer()
    token = server.bind('owner1','j1')
    assert server.bind('owner1','j1') == token
    assert server.bind('owner2','j1') != token
    url=f'http://127.0.0.1:{server.http.server_port}/events/{token}'
    try:
        with urlopen(Request(url,headers={'Origin':'http://localhost:8765'}),timeout=3) as response:
            first=read_event(response)
            assert first['status']=='RUNNING' and 'state' not in first
            state.update(status='COMPLETE',progress={'stage':'COMPLETE','message':'done'})
            _notify_progress()
            assert read_event(response)['status']=='COMPLETE'
        for target, origin, code in [(url,'http://evil.example',403),
                                    (url+'wrong','http://localhost:8765',404)]:
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(target,headers={'Origin':origin}),timeout=2)
            assert exc.value.code == code
    finally:
        server.http.shutdown()
        server.http.server_close()


def test_progress_component_registers_in_real_streamlit(monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from circuitcare.dashboard.components import review_progress
    monkeypatch.setattr(review_progress, '_COMPONENT', None)
    monkeypatch.setattr(review_progress, 'progress_server', lambda: SimpleNamespace(
        bind=lambda *_a:'token', http=SimpleNamespace(server_port=12345)))
    app = AppTest.from_string('''
import streamlit as st
from circuitcare.dashboard.components.review_progress import render_review_progress
render_review_progress(st,owner="owner",job_id="job",job={"status":"RUNNING","progress":{}},on_completed=lambda:None)
''').run()
    assert not app.exception


def test_returned_completion_event_is_applied_once_when_session_value_lags(monkeypatch):
    from types import SimpleNamespace
    from circuitcare.dashboard.components import review_progress
    server=SimpleNamespace(bind=lambda *_a:'token',http=SimpleNamespace(server_port=12345))
    monkeypatch.setattr(review_progress,'progress_server',lambda:server)
    monkeypatch.setattr(review_progress,'_COMPONENT',lambda **_kw:SimpleNamespace(completed={'job_id':'job','token':'token'}))
    st=SimpleNamespace(session_state={})
    commits=[]
    for _ in range(2):
        review_progress.render_review_progress(st,owner='owner',job_id='job',
            job={'status':'COMPLETE'},on_completed=lambda:commits.append('committed'))
    assert commits == ['committed']
    monkeypatch.setattr(review_progress,'_COMPONENT',lambda **_kw:SimpleNamespace(completed={'job_id':'other-job','token':'token'}))
    review_progress.render_review_progress(st,owner='owner',job_id='job',
        job={'status':'COMPLETE'},on_completed=lambda:commits.append('bad'))
    assert commits == ['committed']
