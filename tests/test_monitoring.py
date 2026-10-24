"""Offline semantic metrics/SSE tests; does not load a model or call a provider."""
import time
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from runtime.chatbot import api, llm
from loadtest_support import consume_sse, percentile, StreamValidationError


def sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0


class MetricsTests(unittest.TestCase):
    def setUp(self):
        rag = Mock()
        rag.cache = None
        rag.search.return_value = [{'id':'test','doc_type':'product_card','score':.5,
                                    'text':'synthetic','metadata':{'product_id':1,'name':'test'}}]
        self.factory = patch.object(api, 'TikiRAG', return_value=rag)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.close = patch.object(llm, 'close_clients')
        self.close.start()
        self.addCleanup(self.close.stop)
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_http200_sse_error_is_counted_as_semantic_failure(self):
        before = sample('tiki_stream_outcomes_total', {'outcome':'provider_error'})
        first = sample('tiki_stream_first_token_seconds_count')
        def broken(*args):
            yield 'partial'
            raise llm.LLMStreamError('mock failure')
        with patch.object(llm, 'configured_providers', return_value=['mock']), patch.object(llm, 'ask_stream', side_effect=broken):
            response = self.client.post('/chat/stream', json={'query':'q'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('event: error', response.text)
        self.assertEqual(sample('tiki_stream_outcomes_total', {'outcome':'provider_error'}), before + 1)
        self.assertEqual(sample('tiki_stream_first_token_seconds_count'), first + 1)
        self.assertEqual(sample('tiki_streams_in_progress'), 0)

    def test_completed_stream_duration_includes_token_generation(self):
        labels = {'method':'POST','route':'/chat/stream'}
        before = sample('tiki_http_request_duration_seconds_sum', labels)
        completed = sample('tiki_stream_outcomes_total', {'outcome':'completed'})
        def stream(*args):
            time.sleep(.04)
            yield 'test'
        with patch.object(llm, 'configured_providers', return_value=['mock']), patch.object(llm, 'ask_stream', side_effect=stream):
            response = self.client.post('/chat/stream', json={'query':'q'})
        self.assertIn('event: done', response.text)
        self.assertGreaterEqual(sample('tiki_http_request_duration_seconds_sum', labels) - before, .04)
        self.assertEqual(sample('tiki_stream_outcomes_total', {'outcome':'completed'}), completed + 1)
        self.assertEqual(sample('tiki_http_requests_in_progress'), 0)

    def test_preflight_error_and_unmatched_route_cardinality(self):
        before = sample('tiki_stream_outcomes_total', {'outcome':'preflight_error'})
        with patch.object(llm, 'configured_providers', side_effect=llm.LLMConfigurationError('missing')):
            self.assertEqual(self.client.post('/chat/stream', json={'query':'q'}).status_code, 503)
        self.assertEqual(sample('tiki_stream_outcomes_total', {'outcome':'preflight_error'}), before + 1)
        self.client.get('/a-random-url-123')
        self.assertGreaterEqual(sample('tiki_http_requests_total', {'method':'GET','route':'unmatched','status':'404'}), 1)


class SSEValidationTests(unittest.TestCase):
    def test_complete_and_first_token_callback(self):
        tokens = []
        result = consume_sse([': comment','event: token','data: {"text":"xin chào"}','','event: done','data: {"status":"ok"}',''], tokens.append)
        self.assertEqual(result['token_events'], 1)
        self.assertEqual(tokens, ['xin chào'])

    def test_errors_missing_done_and_empty_completion_fail(self):
        for lines in [
            ['event: error','data: {"detail":"failure"}',''],
            ['event: token','data: {"text":"partial"}',''],
            ['event: done','data: {"status":"ok"}',''],
            ['event: token','data: invalid',''],
        ]:
            with self.subTest(lines=lines), self.assertRaises(StreamValidationError):
                consume_sse(lines)

    def test_percentiles(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([10,20,30], .5), 20)


if __name__ == '__main__':
    unittest.main()
