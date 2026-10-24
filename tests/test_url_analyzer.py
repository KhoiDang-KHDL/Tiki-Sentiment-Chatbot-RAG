"""URL pipeline tests without Tiki requests, Redis, downloads or LLM calls."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runtime.chatbot import url_analyzer as ua


def product():
    return ua.ProductAnalysis(product_id="123", name="Sản phẩm A", price=None,
        rating=None, review_count=None, brand="", category="", url="https://tiki.vn/a-p123.html",
        aspect_scores={"quality": {"pos": 1, "neg": 0}}, total_analyzed=20)


class URLTests(unittest.TestCase):
    def test_only_tiki_urls_and_numeric_ids(self):
        for url in ("https://tiki.vn/a-p123.html?spid=456", "https://www.tiki.vn/a-p123.html", "123"):
            self.assertEqual(ua.extract_product_id(url), "123")
        for url in ("https://evil.vn/a-p123.html", "https://tiki.vn.evil.vn/a-p123.html",
                    "https://user@tiki.vn/a-p123.html", "http://localhost/a-p123.html", "0", "-1", "no ID", "９９"):
            self.assertIsNone(ua.extract_product_id(url))

    def test_report_handles_nulls_and_avoids_false_zero_satisfaction(self):
        item = product()
        item.aspect_scores = {"quality": {"pos": None, "neg": None}}
        item.sample_reviews = [{"rating": "invalid", "content": "Test"}]
        text = ua.format_analysis_report(item)
        self.assertIn("Chưa có giá", text)
        self.assertIn("Chưa có ý kiến", text)
        self.assertNotIn("0% hài lòng", text)
        self.assertNotIn("F1=", text)
        self.assertEqual(ua.analysis_to_document(item)["metadata"]["product_id"], 123)

    def test_kb_collection_counts_and_client_close(self):
        metadata = {"name": "A", "absa_n_reviews": 20, "absa_quality_pos": 3,
                    "absa_quality_neg": 1, "absa_price_pos": None, "absa_price_neg": None}
        client = Mock()
        client.scroll.side_effect = [([NS(payload={"metadata": metadata})], None), ([], None)]
        with patch("qdrant_client.QdrantClient", return_value=client), \
                patch.dict("os.environ", {"QDRANT_COLLECTION": "test_collection"}):
            result = ua._check_qdrant_kb("123")
        self.assertEqual(result.total_analyzed, 20)
        self.assertEqual(result.aspect_scores["quality"]["pct"], 75)
        self.assertIsNone(result.aspect_scores["price"]["pct"])
        self.assertTrue(all(call.kwargs["collection_name"] == "test_collection" for call in client.scroll.call_args_list))
        client.close.assert_called_once()

    def test_reviews_deduplicate_and_limit_pages(self):
        page = [{"id": 1, "content": "great", "rating": 5}, {"id": 1, "content": "great"}, {"content": " "}]
        with patch("runtime.crawling.module3_product_reviews.fetch_reviews_page", side_effect=[(page, 2), (page, 2)]) as fetch:
            reviews = ua.crawl_reviews_fast("123")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(fetch.call_count, 2)

    def test_http_block_is_reported_without_bypass(self):
        response = Mock(status_code=403)
        with patch("requests.get", return_value=response) as get:
            with self.assertRaisesRegex(ua.AnalysisError, "HTTP 403"):
                ua.fetch_product_detail("123")
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertEqual(get.call_count, 1)


class AsyncURLTests(unittest.IsolatedAsyncioTestCase):
    async def test_kb_hit_does_not_crawl_or_infer(self):
        with patch.object(ua, "_cache", return_value=None), patch.object(ua, "_check_qdrant_kb", return_value=product()), \
                patch.object(ua, "fetch_product_detail") as fetch, patch.object(ua, "_infer") as infer:
            result = await ua.analyze_product("123", absa_model="phobert")
        self.assertEqual(result.source, "kb")
        fetch.assert_not_called()
        infer.assert_not_called()

    async def test_cache_preserves_model_and_scopes_key(self):
        cached = asdict(product())
        with patch.object(ua, "_cache", return_value=cached) as cache, patch.object(ua, "_check_qdrant_kb") as kb:
            result = await ua.analyze_product("123", absa_model="logreg")
        self.assertEqual(result.source, "cache")
        self.assertIn(":logreg:123", cache.call_args.args[1])
        kb.assert_not_called()

    async def test_live_inference_uses_selected_model_off_event_loop(self):
        import threading
        event_thread = threading.get_ident()
        def infer(reviews, model):
            self.assertNotEqual(threading.get_ident(), event_thread)
            self.assertEqual(model, "phobert_onnx")
            return {"quality": {"pos": 1, "neg": 0}}
        with patch.object(ua, "_cache"), patch.object(ua, "_check_qdrant_kb") as kb, \
                patch.object(ua, "fetch_product_detail", return_value={"name": "A", "categories": [{"name": "Category"}]}), \
                patch.object(ua, "crawl_reviews_fast", return_value=[{"rating": 5, "content": "Great"}]), \
                patch.object(ua, "_infer", side_effect=infer):
            result = await ua.analyze_product("123", absa_model="phobert_onnx", force_refresh=True)
        kb.assert_not_called()
        self.assertEqual(result.total_analyzed, 1)
        self.assertEqual(result.category, "Category")

    async def test_reviews_failure_keeps_report_but_marks_missing_absa(self):
        with patch.object(ua, "_cache") as cache, patch.object(ua, "fetch_product_detail", return_value={"name": "A"}), \
                patch.object(ua, "crawl_reviews_fast", side_effect=ua.AnalysisError("blocked")):
            result = await ua.analyze_product("123", force_refresh=True)
        self.assertEqual(result.total_analyzed, 0)
        self.assertTrue(result.warnings)
        cache.assert_not_called()

    async def test_missing_model_has_actionable_error(self):
        with patch.object(ua, "fetch_product_detail", return_value={"name": "A"}), \
                patch.object(ua, "crawl_reviews_fast", return_value=[{"content": "A"}]), \
                patch.object(ua, "_infer", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(ua.AnalysisError, "checkpoint"):
                await ua.analyze_product("123", force_refresh=True)

    async def test_llm_failure_keeps_deterministic_report(self):
        from runtime.chatbot import llm
        with patch.object(ua, "analyze_product", return_value=product()), \
                patch.object(llm, "ask_url_recommendation", side_effect=llm.LLMError("HTTP 429")):
            text = await ua.analyze_url("123", user_query="Có nên mua?")
        self.assertIn("Sản phẩm A", text)
        self.assertIn("429", text)

    async def test_bad_url_or_model_rejected_before_io(self):
        with patch.object(ua, "_cache") as cache:
            with self.assertRaises(ValueError):
                await ua.analyze_product("https://evil.vn/a-p123.html")
            with self.assertRaises(ValueError):
                await ua.analyze_product("123", absa_model="invalid")
        cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
