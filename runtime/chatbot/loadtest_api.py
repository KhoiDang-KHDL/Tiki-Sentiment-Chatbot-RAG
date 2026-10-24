"""Isolated synthetic load-test server. Never enabled inside the normal API.

Dedicated Compose service has no provider keys, model mounts, or Qdrant network.
"""
import os
from types import SimpleNamespace
import time

if os.getenv('TIKI_LOADTEST_MODE') != 'synthetic':
    raise RuntimeError('This entrypoint requires TIKI_LOADTEST_MODE=synthetic')

from runtime.chatbot import api, llm

DOC = {'id': 'synthetic_product', 'doc_type': 'product_card', 'score': 0.9,
       'text': 'Dữ liệu giả lập chỉ dùng kiểm thử tải. Sản phẩm thử có chất lượng tốt.',
       'metadata': {'product_id': 1, 'name': 'Sản phẩm giả lập kiểm thử tải'}}


class SyntheticRAG:
    def __init__(self):
        self.cache = None
        self.client = SimpleNamespace(get_collection=lambda name: SimpleNamespace(points_count=1))
    def search(self, query, **kwargs):
        time.sleep(0.01)
        return [DOC]
    def close(self):
        pass


def ask_stream(query, docs, history=None):
    time.sleep(0.05)
    for part in ['Đây ', 'là ', 'câu trả lời ', 'giả lập ', '[Nguồn 1].']:
        time.sleep(0.02)
        yield part
        if query == '__stream_error__':
            raise llm.LLMStreamError('Synthetic injected stream error')


api.TikiRAG = SyntheticRAG
llm.rewrite_query = lambda query, history: query
llm.ask = lambda query, docs, history=None: ''.join(ask_stream(query, docs, history))
llm.ask_stream = ask_stream
llm.configured_providers = lambda: ['synthetic-no-external-provider']
llm.close_clients = lambda: None
app = api.app


@app.middleware('http')
async def mark_synthetic(request, call_next):
    response = await call_next(request)
    response.headers['X-Tiki-Loadtest'] = 'synthetic'
    return response
