"""Map the existing local client contract to native mlx-vlm OpenAI endpoints."""
import os
import time


def request_mlx(transport, method, path, payload=None):
    model = os.environ.get('CIRCUITCARE_MLX_MODEL', 'mlx-community/Qwen3.5-9B-MLX-4bit')
    if path in {'/api/tags', '/api/show'}:
        result = transport('GET', '/v1/models')
        available = any(row.get('id') == model for row in result.get('data', []))
        if not available:
            raise OSError(f'MLX model is not loaded: {model}')
        if path == '/api/show':
            return {'capabilities': ['completion', 'vision'], 'backend': 'mlx-vlm', 'model': model}
        return {'models': [{'name': 'qwen3.5:9b', 'model': 'qwen3.5:9b',
                            'backend': 'mlx-vlm', 'native_model': model}]}
    if path != '/api/chat' or method != 'POST' or not isinstance(payload, dict):
        raise ValueError(f'Unsupported MLX operation: {method} {path}')
    if payload.get('model') not in {'qwen3.5:9b', model}:
        raise ValueError('The native MLX profile only supports the configured Qwen model.')
    messages = []
    for original in payload.get('messages', []):
        message = {key: value for key, value in original.items() if key != 'images'}
        if original.get('images'):
            message['content'] = [{'type': 'text', 'text': original.get('content', '')},
                *[{'type': 'image_url', 'image_url': {'url':
                    ('data:image/jpeg;base64,' if image.startswith('/9j/') else 'data:image/png;base64,') + image}}
                  for image in original['images']]]
        messages.append(message)
    options = payload.get('options', {})
    body = {'model': model, 'messages': messages, 'stream': False, 'enable_thinking': False,
            'max_tokens': options.get('num_predict', 2048), 'temperature': options.get('temperature', .1)}
    schema = payload.get('format')
    if isinstance(schema, dict):
        body['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'circuitcare_result', 'schema': schema, 'strict': True}}
    elif schema == 'json':
        body['response_format'] = {'type': 'json_object'}
    started = time.perf_counter_ns()
    result = transport('POST', '/v1/chat/completions', body)
    choice = (result.get('choices') or [{}])[0]
    usage = result.get('usage') or {}
    return {'message': choice.get('message', {}),
            'done_reason': choice.get('finish_reason'), 'total_duration': time.perf_counter_ns() - started,
            'prompt_eval_count': usage.get('prompt_tokens'), 'eval_count': usage.get('completion_tokens'),
            'backend': 'mlx-vlm'}
