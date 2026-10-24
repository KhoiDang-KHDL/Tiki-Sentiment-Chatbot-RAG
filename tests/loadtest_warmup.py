"""Explicit, unmeasured retrieval warm-up. No chat/provider calls."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import urllib.request
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--requests', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if urlparse(args.host).hostname not in {'api', 'api-loadtest'} or not 0 <= args.requests <= 20:
        parser.error('Warm-up is limited to the internal API services and at most 20 requests')
    times = []
    for _ in range(args.requests):
        start = perf_counter()
        request = urllib.request.Request(args.host.rstrip('/') + '/search',
            data=json.dumps({'query':'sản phẩm chất lượng tốt','top_k':3}).encode('utf-8'),
            headers={'Content-Type':'application/json'})
        result = json.load(urllib.request.urlopen(request, timeout=120))
        if not result.get('documents'):
            raise RuntimeError('Warm-up search returned no documents')
        times.append((perf_counter() - start) * 1000)
    report = {'host':args.host, 'unmeasured_warmup_requests':args.requests, 'latency_ms':times}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
