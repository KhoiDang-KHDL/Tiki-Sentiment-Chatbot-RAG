"""Low-cardinality HTTP and semantic SSE metrics (no queries or product labels)."""
from time import perf_counter
from prometheus_client import Counter, Gauge, Histogram

BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300)
HTTP_REQUESTS = Counter('tiki_http_requests_total', 'Completed HTTP requests', ['method', 'route', 'status'])
HTTP_DURATION = Histogram('tiki_http_request_duration_seconds', 'Entire HTTP response including streaming body',
                          ['method', 'route'], buckets=BUCKETS)
HTTP_ACTIVE = Gauge('tiki_http_requests_in_progress', 'Active HTTP requests excluding health and metrics')
STREAM_OUTCOMES = Counter('tiki_stream_outcomes_total', 'Semantic stream outcomes, including HTTP 200 error events', ['outcome'])
STREAM_ACTIVE = Gauge('tiki_streams_in_progress', 'Streaming response generators in progress')
STREAM_TTFT = Histogram('tiki_stream_first_token_seconds', 'Handler start to first nonempty token, including retrieval/queue', buckets=BUCKETS)
STREAM_DURATION = Histogram('tiki_stream_duration_seconds', 'Handler start to stream finish', ['outcome'], buckets=BUCKETS)


def stream_preflight_error(started):
    STREAM_OUTCOMES.labels('preflight_error').inc()
    STREAM_DURATION.labels('preflight_error').observe(perf_counter() - started)


class StreamObservation:
    def __init__(self, started):
        self.started, self.first, self.finished = started, False, False
        STREAM_ACTIVE.inc()

    def token(self, text):
        if text and not self.first:
            STREAM_TTFT.observe(perf_counter() - self.started)
            self.first = True

    def finish(self, outcome):
        if not self.finished:
            STREAM_OUTCOMES.labels(outcome).inc()
            STREAM_DURATION.labels(outcome).observe(perf_counter() - self.started)
            STREAM_ACTIVE.dec()
            self.finished = True


class MonitoringMiddleware:
    """ASGI middleware observes completion/cancellation, not just response headers."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') in {'/health', '/metrics'}:
            return await self.app(scope, receive, send)
        started, status = perf_counter(), 500
        HTTP_ACTIVE.inc()
        async def observed_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
            await send(message)
        try:
            await self.app(scope, receive, observed_send)
        finally:
            route = getattr(scope.get('route'), 'path', 'unmatched')
            HTTP_REQUESTS.labels(scope['method'], route, str(status)).inc()
            HTTP_DURATION.labels(scope['method'], route).observe(perf_counter() - started)
            HTTP_ACTIVE.dec()
