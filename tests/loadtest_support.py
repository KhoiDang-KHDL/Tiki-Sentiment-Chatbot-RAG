"""Pure SSE validation shared by Locust and offline unit tests."""
import json


class StreamValidationError(RuntimeError):
    pass


def consume_sse(lines, on_token=None):
    event, payload, tokens = 'message', [], 0
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if line:
            if line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                payload.append(line[5:].lstrip())
            continue
        if not payload:
            event = 'message'
            continue
        try:
            data = json.loads('\n'.join(payload))
        except ValueError as exc:
            raise StreamValidationError('Invalid SSE JSON') from exc
        if event == 'error':
            # Do not put raw provider response text into Locust's error labels.
            raise StreamValidationError('SSE error event (HTTP may still be 200)')
        if event == 'token':
            if not isinstance(data, dict) or not isinstance(data.get('text'), str):
                raise StreamValidationError('Invalid token event')
            if data['text']:
                tokens += 1
                if on_token:
                    on_token(data['text'])
        if event == 'done':
            if not isinstance(data, dict) or data.get('status') != 'ok':
                raise StreamValidationError('Invalid completion status')
            if tokens == 0:
                raise StreamValidationError('Completed stream without tokens')
            return {'token_events': tokens}
        event, payload = 'message', []
    raise StreamValidationError('Stream closed before done')


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
