"""HTTP integration checks; refuses to test a non-synthetic API."""
import argparse
import json
import time
import urllib.error
import urllib.request


def request(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.headers, response.read().decode('utf-8')


def wait(url):
    deadline = time.monotonic() + 90
    while True:
        try:
            return request(url)
        except (urllib.error.URLError, TimeoutError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://127.0.0.1:18000')
    parser.add_argument('--ui', default='http://127.0.0.1:18001')
    args = parser.parse_args()
    headers, _ = wait(args.api + '/health')
    if headers.get('X-Tiki-Loadtest') != 'synthetic':
        raise RuntimeError('Refusing chat smoke test against a real API')
    _, body = request(args.api + '/search', {'query': 'synthetic'})
    if not json.loads(body)['documents']:
        raise AssertionError('Empty retrieval result')
    _, body = request(args.api + '/chat', {'query': 'synthetic'})
    if not json.loads(body)['answer']:
        raise AssertionError('Empty answer')
    _, stream = request(args.api + '/chat/stream', {'query': 'synthetic'})
    if 'event: token' not in stream or 'event: done' not in stream or 'event: error' in stream:
        raise AssertionError('Incomplete successful SSE stream')
    _, error = request(args.api + '/chat/stream', {'query': '__stream_error__'})
    if 'event: error' not in error or 'event: done' in error:
        raise AssertionError('Injected failure was not represented as an SSE error')
    _, metrics = request(args.api + '/metrics')
    if 'tiki_http_requests_total' not in metrics or 'outcome="provider_error"' not in metrics:
        raise AssertionError('Missing request or semantic stream error metrics')
    wait(args.ui + '/')
    print('PASS: synthetic health/search/chat/SSE success+error/metrics; HTTP UI reachable.')


if __name__ == '__main__':
    main()
