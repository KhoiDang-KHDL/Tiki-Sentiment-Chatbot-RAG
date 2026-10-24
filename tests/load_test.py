"""Locust scenarios. Default host is the isolated synthetic API, never a provider.

HTTP response_time for SSE is overwritten with total body consumption time.
First-token samples go to summary.json, not duplicate pseudo-HTTP requests.
"""
import json
import csv
import os
from pathlib import Path
import random
from time import perf_counter

from locust import HttpUser, between, events, task
from locust.exception import StopUser
from loadtest_support import consume_sse, percentile, StreamValidationError

QUERIES = ['sản phẩm chất lượng tốt', 'đóng gói cẩn thận', 'giao hàng nhanh',
           'dụng cụ nhà bếp', 'sản phẩm giá hợp lý']
FIRST_TOKENS = []
STARTED = None


class BaseUser(HttpUser):
    abstract = True
    wait_time = between(0.3, 0.8)
    uses_llm = False

    def on_start(self):
        with self.client.get('/health', name='/health [start]', catch_response=True, timeout=15) as response:
            self.synthetic = response.headers.get('X-Tiki-Loadtest') == 'synthetic'
            if response.status_code != 200:
                response.failure('Target health check failed')
                raise StopUser()
        if self.uses_llm and not self.synthetic and os.getenv('ALLOW_LIVE_LLM') != '1':
            self.environment.tiki_live_gate_refused = True
            raise StopUser('Live LLM calls disabled; use synthetic target or explicit ALLOW_LIVE_LLM=1')

    def health(self):
        with self.client.get('/health', catch_response=True, timeout=15) as response:
            if response.status_code == 200:
                try:
                    if response.json().get('status') != 'ok':
                        response.failure('Health status is not ok')
                except ValueError:
                    response.failure('Invalid health JSON')

    def search(self):
        with self.client.post('/search', json={'query':random.choice(QUERIES), 'top_k':3},
                              catch_response=True, timeout=120) as response:
            if response.status_code == 200:
                try:
                    if not response.json().get('documents'):
                        response.failure('Search returned no documents')
                except ValueError:
                    response.failure('Invalid search JSON')

    def chat(self):
        with self.client.post('/chat', json={'query':random.choice(QUERIES), 'top_k':3},
                              catch_response=True, timeout=180) as response:
            if response.status_code == 200:
                try:
                    if not response.json().get('answer'):
                        response.failure('Empty answer')
                except ValueError:
                    response.failure('Invalid chat JSON')

    def stream(self):
        started = perf_counter()
        first = []
        body_length = 0
        def token(text):
            if not first:
                first.append((perf_counter() - started) * 1000)
        with self.client.post('/chat/stream', json={'query':random.choice(QUERIES), 'top_k':3},
                              stream=True, catch_response=True, timeout=180) as response:
            try:
                if response.status_code != 200:
                    response.failure(f'HTTP {response.status_code}')
                    return
                def lines():
                    nonlocal body_length
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        body_length += len(line.encode('utf-8')) + 1
                        yield line
                consume_sse(lines(), on_token=token)
                response.success()
                if first:
                    FIRST_TOKENS.append(first[0])
            except Exception as exc:
                # Any interrupted/bad SSE is failure even if HTTP headers were 200.
                response.failure(str(exc) if isinstance(exc, StreamValidationError) else type(exc).__name__)
            finally:
                response.request_meta['response_time'] = (perf_counter() - started) * 1000
                response.request_meta['response_length'] = body_length
                response.close()


class HealthUser(BaseUser):
    @task
    def request_health(self):
        self.health()


class SearchUser(BaseUser):
    @task
    def request_search(self):
        self.search()


class ChatUser(BaseUser):
    uses_llm = True
    @task
    def request_chat(self):
        self.chat()


class StreamingUser(BaseUser):
    uses_llm = True
    @task
    def request_stream(self):
        self.stream()


class MixedUser(BaseUser):
    uses_llm = True
    @task(1)
    def request_health(self):
        self.health()
    @task(3)
    def request_search(self):
        self.search()
    @task(2)
    def request_chat(self):
        self.chat()
    @task(3)
    def request_stream(self):
        self.stream()


@events.test_start.add_listener
def start_test(environment, **kwargs):
    global STARTED
    STARTED = perf_counter()
    FIRST_TOKENS.clear()
    environment.tiki_live_gate_refused = False


@events.test_stop.add_listener
def finish_test(environment, **kwargs):
    total = environment.stats.total
    threshold_ms = float(os.getenv('LOADTEST_P95_MS', '10000'))
    threshold_fail = float(os.getenv('LOADTEST_MAX_FAILURE_RATIO', '0.01'))
    p95 = total.get_response_time_percentile(0.95) if total.num_requests else None
    rows = []
    for entry in environment.stats.entries.values():
        rows.append({'method':entry.method, 'name':entry.name, 'requests':entry.num_requests,
                     'failures':entry.num_failures, 'avg_ms':entry.avg_response_time,
                     'p50_ms':entry.get_response_time_percentile(0.5),
                     'p95_ms':entry.get_response_time_percentile(0.95)})
    result = {'mode':os.getenv('LOADTEST_MODE', 'interactive'), 'scenario':os.getenv('LOADTEST_SCENARIO', 'interactive'),
              'host':environment.host, 'users':getattr(environment.parsed_options, 'num_users', None),
              'duration_seconds':perf_counter() - STARTED if STARTED else None,
              'warmup_requests':int(os.getenv('LOADTEST_WARMUP_REQUESTS','0')),
              'requests':total.num_requests, 'failures':total.num_failures,
              'failure_ratio':total.fail_ratio, 'requests_per_second':total.total_rps,
              'p50_ms':total.get_response_time_percentile(0.5) if total.num_requests else None,
              'p95_ms':p95, 'ttft_samples':len(FIRST_TOKENS),
              'ttft_p50_ms':percentile(FIRST_TOKENS, .5), 'ttft_p95_ms':percentile(FIRST_TOKENS, .95),
              'thresholds':{'p95_ms':threshold_ms, 'max_failure_ratio':threshold_fail},
              'endpoints':rows,
              'note':'Synthetic mode does not measure actual embedding or external LLM capacity. First-token samples include completed streams only.'}
    result['live_llm_blocked'] = getattr(environment, 'tiki_live_gate_refused', False)
    result['passed'] = bool(total.num_requests and not result['live_llm_blocked'] and total.fail_ratio <= threshold_fail and p95 <= threshold_ms)
    path = Path(os.getenv('LOADTEST_SUMMARY', '/reports/interactive-summary.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # Final CSV snapshot avoids the background CSV writer shutdown race in the
    # official Locust 2.47/Python 3.14 image.
    with path.with_name('requests.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method','name','requests','failures','avg_ms','p50_ms','p95_ms'])
        writer.writeheader()
        writer.writerows(rows)
    environment.tiki_loadtest_passed = result['passed']


@events.quitting.add_listener
def check_thresholds(environment, **kwargs):
    if getattr(environment, 'tiki_loadtest_passed', True) is False:
        environment.process_exit_code = 1
