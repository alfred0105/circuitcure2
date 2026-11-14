import pytest
from circuitcare.mlx_transport import request_mlx


def test_native_chat_keeps_images_schema_and_disables_thinking():
    calls = []
    def send(method, path, body=None):
        calls.append((method, path, body))
        return {'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 4}}
    schema = {'type': 'object'}
    response = request_mlx(send, 'POST', '/api/chat', {
        'model': 'qwen3.5:9b', 'messages': [{'role': 'user', 'content': 'inspect', 'images': ['aGVsbG8=']}],
        'format': schema, 'options': {'num_predict': 300, 'temperature': .1}})
    body = calls[0][2]
    assert calls[0][:2] == ('POST', '/v1/chat/completions')
    assert body['enable_thinking'] is False
    assert body['max_tokens'] == 300
    assert body['messages'][0]['content'][1]['image_url']['url'].endswith('aGVsbG8=')
    assert body['response_format']['json_schema']['schema'] == schema
    assert response['message']['content'] == '{"ok":true}'
    assert response['eval_count'] == 4


def test_native_metadata_requires_configured_model_loaded():
    with pytest.raises(OSError):
        request_mlx(lambda *_a: {'data': [{'id': 'wrong-model'}]}, 'GET', '/api/tags')
    tags = request_mlx(lambda *_a: {'data': [{'id': 'mlx-community/Qwen3.5-9B-4bit'}]}, 'GET', '/api/tags')
    assert tags['models'][0]['name'] == 'qwen3.5:9b'
