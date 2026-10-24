"""HTTP UI and SSE transport tests; no external services or model execution."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from runtime.chatbot import api_client, demo_chainlit_http as ui


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_posts_history_and_handles_done(self):
        seen = []
        def handler(request):
            import json
            seen.append(json.loads(request.content))
            return httpx.Response(200, text='event: sources\ndata: []\n\nevent: token\ndata: {"text":"xin chào"}\n\nevent: done\ndata: {}\n\n')
        client_class = httpx.AsyncClient
        with patch.object(api_client.httpx, 'AsyncClient', side_effect=lambda **kw: client_class(transport=httpx.MockTransport(handler), **kw)):
            events = [e async for e in api_client.stream_chat('q', [{'role':'user','content':'old'}], context_id='ctx')]
        self.assertEqual(events[1], ('token', {'text':'xin chào'}))
        self.assertEqual(seen[0]['context_id'], 'ctx')
        self.assertEqual(seen[0]['history'][0]['content'], 'old')

    async def test_missing_done_error_event_and_http_failure(self):
        responses = [httpx.Response(200, text='event: token\ndata: {"text":"partial"}\n\n'),
                     httpx.Response(200, text='event: error\ndata: {"detail":"quota"}\n\n'),
                     httpx.Response(503, json={'detail':'API unavailable'})]
        client_class = httpx.AsyncClient
        for response in responses:
            with self.subTest(response=response.status_code):
                with patch.object(api_client.httpx, 'AsyncClient', side_effect=lambda **kw: client_class(transport=httpx.MockTransport(lambda r: response), **kw)):
                    with self.assertRaises(api_client.APIError):
                        _ = [e async for e in api_client.stream_chat('q', [])]

    def test_credentials_not_allowed_in_api_url(self):
        with patch.dict(os.environ, {'API_BASE_URL':'http://user:password@host'}):
            with self.assertRaises(api_client.APIError):
                api_client.base_url()


class Session:
    def __init__(self):
        self.values = {'history':[], 'context_id':None, 'absa_model':'logreg'}
    def get(self, key, default=None):
        return self.values.get(key, default)
    def set(self, key, value):
        self.values[key] = value


class Message:
    sent = []
    def __init__(self, content=''):
        self.content = content
    async def send(self):
        self.sent.append(self)
        return self
    async def update(self):
        return self
    async def stream_token(self, text):
        self.content += text


class HTTPUITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Session()
        Message.sent = []
        self.fake = patch.object(ui, 'cl', SimpleNamespace(user_session=self.session, Message=Message))
        self.fake.start()
        self.addCleanup(self.fake.stop)

    async def test_chat_uses_http_and_saves_only_complete_turn(self):
        async def stream(*args, **kwargs):
            yield 'sources', []
            yield 'token', {'text':'answer'}
            yield 'done', {}
        with patch.object(api_client, 'stream_chat', side_effect=stream) as remote:
            await ui.handle_chat('question')
        remote.assert_called_once()
        self.assertEqual(self.session.get('history')[-1]['content'], 'answer')
        self.assertFalse(hasattr(ui, 'TikiRAG'))

    async def test_partial_answer_not_saved(self):
        async def stream(*args, **kwargs):
            yield 'sources', []
            yield 'token', {'text':'partial'}
            raise api_client.APIError('interrupted')
        with patch.object(api_client, 'stream_chat', side_effect=stream):
            await ui.handle_chat('question')
        self.assertEqual(self.session.get('history'), [])
        self.assertIn('partial', Message.sent[0].content)

    async def test_url_report_uses_api_without_chat_and_failed_url_clears_context(self):
        with patch.object(api_client, 'analyze', return_value={'context_id':'abc','report':'Report'}) as remote, \
                patch.object(ui, 'handle_chat') as chat:
            await ui.handle_url_analysis('123', '123')
        self.assertEqual(self.session.get('context_id'), 'abc')
        chat.assert_not_called()
        with patch.object(api_client, 'analyze', side_effect=api_client.APIError('failed')):
            await ui.handle_url_analysis('456')
        self.assertIsNone(self.session.get('context_id'))


if __name__ == '__main__':
    unittest.main()
