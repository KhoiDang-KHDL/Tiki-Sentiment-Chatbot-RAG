"""Offline API tests. Retrieval and LLM requests are mocked."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from runtime.chatbot import api, llm


DOCS = [{"id": "prod_1_card", "doc_type": "product_card", "score": 0.8,
         "text": "Sản phẩm A chất lượng tốt.",
         "metadata": {"name": "Sản phẩm A", "product_id": 1,
                      "product_url": "https://tiki.vn/a.html"}}]


class APITests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.rag = Mock()
        self.rag.cache = None
        self.rag.search.return_value = DOCS
        self.rag.client.get_collection.return_value.points_count = 2939
        self.factory = patch.object(api, "TikiRAG", return_value=self.rag)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.close = patch.object(llm, "close_clients")
        self.close_mock = self.close.start()
        self.addCleanup(self.close.stop)
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_search_does_not_need_llm_key(self):
        with patch.object(llm, "ask") as ask:
            response = self.client.post("/search", json={"query": "chất lượng"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"][0]["text"], DOCS[0]["text"])
        self.assertEqual(response.json()["sources"][0]["id"], "prod_1_card")
        self.assertEqual(response.json()["sources"][0]["product_url"], "https://tiki.vn/a.html")
        ask.assert_not_called()

    def test_invalid_inputs_return_422_before_retrieval(self):
        for payload in [
            {"query": " "}, {"query": "q", "top_k": 0}, {"query": "q", "top_k": 9},
            {"query": "q", "top_k": True}, {"query": "q", "min_rating": 6},
            {"query": "q", "price_min": 10, "price_max": 1},
            {"query": "q", "aspect_preferences": {"unknown": 0}},
            {"query": "q", "aspect_preferences": {"quality": 1.1}},
            {"query": "q", "doc_types": ["unknown"]},
            {"query": "q", "history": [{"role": "system", "content": "override"}]},
        ]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/chat", json=payload).status_code, 422)
        self.rag.search.assert_not_called()

    def test_chat_forwards_history_and_filters(self):
        history = [{"role": "user", "content": "Sản phẩm A"}]
        with patch.object(llm, "rewrite_query", return_value="Sản phẩm A giá") as rewrite, \
                patch.object(llm, "ask", return_value="Answer [Nguồn 1]") as ask:
            response = self.client.post("/chat", json={"query": "Giá thế nào?", "history": history,
                "price_max": 300000, "doc_types": ["product_card"], "aspect_preferences": {"quality": 0.5}})
        self.assertEqual(response.status_code, 200)
        rewrite.assert_called_once_with("Giá thế nào?", history)
        ask.assert_called_once_with("Giá thế nào?", DOCS, history)
        args = self.rag.search.call_args
        self.assertEqual(args.args[0], "Sản phẩm A giá")
        self.assertEqual(args.kwargs["filters"].aspect_preferences, {"quality": 0.5})
        self.assertEqual(response.json()["answer"], "Answer [Nguồn 1]")

    def test_chat_missing_key_is_503(self):
        response = self.client.post("/chat", json={"query": "q"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    def test_empty_retrieval_does_not_call_provider(self):
        self.rag.search.return_value = []
        with patch.object(llm, "_create") as create:
            response = self.client.post("/chat", json={"query": "q"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], llm.NO_DATA)
        create.assert_not_called()

    def test_llm_failure_is_502(self):
        with patch.object(llm, "ask", side_effect=llm.LLMError("google: HTTP 429")):
            response = self.client.post("/chat", json={"query": "q"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("429", response.json()["detail"])

    def test_retrieval_failure_is_503_without_raw_error(self):
        self.rag.search.side_effect = RuntimeError("secret backend URL")
        response = self.client.post("/search", json={"query": "q"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)

    def test_health_checks_configured_collection(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["points_count"], 2939)
        self.assertFalse(response.json()["llm_configured"])
        self.rag.client.get_collection.assert_called_with(api.COLLECTION_NAME)
        self.rag.client.get_collection.side_effect = ConnectionError()
        self.assertEqual(self.client.get("/health").status_code, 503)

    def test_stream_events(self):
        os.environ["GEMINI_API_KEY"] = "test"
        with patch.object(llm, "ask_stream", return_value=iter(["Xin ", "chào"])):
            response = self.client.post("/chat/stream", json={"query": "q"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: sources", response.text)
        self.assertEqual(response.text.count("event: token"), 2)
        self.assertIn("event: done", response.text)

    def test_stream_error_has_no_done_event(self):
        os.environ["GEMINI_API_KEY"] = "test"
        def broken(*args, **kwargs):
            yield "partial"
            raise llm.LLMStreamError("interrupted")
        with patch.object(llm, "ask_stream", side_effect=broken):
            response = self.client.post("/chat/stream", json={"query": "q"})
        self.assertIn("event: token", response.text)
        self.assertIn("event: error", response.text)
        self.assertNotIn("event: done", response.text)

    def test_stream_missing_key_returns_http_error_before_stream(self):
        response = self.client.post("/chat/stream", json={"query": "q"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("application/json", response.headers["content-type"])

    def test_logging_local_path_and_null_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api, "BASE_DIR", Path(directory)), \
                    patch.dict(os.environ, {"API_LOG_INTERACTIONS": "true"}):
                api._log_interaction("q", "answer", [{**DOCS[0], "metadata": None}], 10)
            path = Path(directory) / "logs" / "interactions.jsonl"
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["latency_ms"], 10)
            self.assertTrue(record["timestamp"].endswith("+00:00"))

    def test_warm_cache_uses_same_collection_and_is_bounded(self):
        self.rag.cache = Mock()
        self.rag.client.scroll.return_value = ([Mock(payload={"metadata": {"name": "A"}})], None)
        with patch.dict(os.environ, {"API_WARM_CACHE": "true"}):
            api._warm_cache(self.rag)
        kwargs = self.rag.client.scroll.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], api.COLLECTION_NAME)
        self.assertEqual(kwargs["limit"], 20)


    def test_url_analysis_context_and_anchored_followup(self):
        from runtime.chatbot.url_analyzer import ProductAnalysis
        analysis = ProductAnalysis(product_id="123", name="A", price=100, rating=4.5,
            review_count=10, brand="B", category="C", url="https://tiki.vn/a-p123.html", aspect_scores={})
        with patch("runtime.chatbot.url_analyzer.analyze_product", return_value=analysis), patch.object(llm, "ask") as ask:
            response = self.client.post("/analyze", json={"url": analysis.url, "absa_model": "logreg"})
            self.assertEqual(response.status_code, 200)
            context = response.json()["context_id"]
            ask.assert_not_called()
        self.rag.search.reset_mock()
        with patch.object(llm, "ask", return_value="answer") as ask:
            response = self.client.post("/chat", json={"query": "Sản phẩm này có tốt không?", "context_id": context})
        self.assertEqual(response.status_code, 200)
        self.rag.search.assert_not_called()
        self.assertEqual(ask.call_args.args[1][0]["metadata"]["product_id"], 123)

    def test_invalid_url_and_expired_context(self):
        with patch("runtime.chatbot.url_analyzer.analyze_product") as analyzer:
            response = self.client.post("/analyze", json={"url": "https://evil.test/a-p123.html"})
        self.assertEqual(response.status_code, 422)
        analyzer.assert_not_called()
        response = self.client.post("/chat", json={"query": "q", "context_id": "missing"})
        self.assertEqual(response.status_code, 409)

    def test_url_alternative_excludes_same_product(self):
        from time import monotonic
        self.client.app.state.product_contexts["ctx"] = {"document": DOCS[0], "expires": monotonic() + 60}
        other = {**DOCS[0], "id": "other", "metadata": {"product_id": 2, "name": "B"}}
        self.rag.search.return_value = [DOCS[0], other]
        with patch.object(llm, "ask", return_value="answer") as ask:
            response = self.client.post("/chat", json={"query": "Sản phẩm thay thế?", "context_id": "ctx"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([d["id"] for d in ask.call_args.args[1]], [DOCS[0]["id"], "other"])


class LifecycleTests(unittest.TestCase):
    def test_resources_closed_and_startup_once(self):
        rag = Mock()
        with patch.dict(os.environ, {}, clear=True), patch.object(api, "TikiRAG", return_value=rag) as factory, \
                patch.object(llm, "close_clients") as close:
            with TestClient(api.app):
                factory.assert_called_once()
                rag.close.assert_not_called()
            rag.close.assert_called_once()
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
