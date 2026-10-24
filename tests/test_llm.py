"""Offline regression tests: no provider requests, model downloads or Qdrant writes."""
import contextlib
import io
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from runtime.chatbot import llm, demo_chat


DOCS = [{"id": "prod_1_card", "doc_type": "product_card", "text": "Thông tin sản phẩm.",
         "metadata": {"product_id": 1, "name": "Sản phẩm A", "price": 100,
                      "product_url": "https://tiki.vn/a.html"}}]


def response(content):
    return NS(text=content, choices=[NS(message=NS(content=content))])


def chunk(content):
    return NS(text=content, choices=[NS(delta=NS(content=content), finish_reason=None)])


class Stream:
    def __init__(self, *items):
        self.items = items
        self.closed = False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def providers(self):
        os.environ.update(LLM_PROVIDER="auto", GROQ_API_KEY="test-groq", GEMINI_API_KEY="test-google")

    def test_missing_keys_and_empty_data(self):
        with self.assertRaises(llm.LLMConfigurationError):
            llm.configured_providers()
        self.assertEqual(llm.ask("Có gì?", []), llm.NO_DATA)
        self.assertEqual(list(llm.ask_stream("Có gì?", [])), [llm.NO_DATA])
        self.assertEqual(llm.rewrite_query("Câu hỏi", []), "Câu hỏi")

    def test_provider_selection(self):
        os.environ["GEMINI_API_KEY"] = "test"
        self.assertEqual(llm.configured_providers(), ["google"])
        os.environ["LLM_PROVIDER"] = "groq"
        with self.assertRaises(llm.LLMConfigurationError):
            llm.configured_providers()

    def test_google_only_does_not_call_groq(self):
        os.environ["GEMINI_API_KEY"] = "test"
        with patch.object(llm, "_create", return_value=response("Trả lời")) as create:
            self.assertEqual(llm.ask("Hỏi", DOCS), "Trả lời")
            self.assertEqual(create.call_args.args[0], "google")

    def test_fallback_and_both_fail(self):
        self.providers()
        with patch.object(llm, "_create", side_effect=[RuntimeError(), response("OK")]) as create:
            self.assertEqual(llm.ask("Hỏi", DOCS), "OK")
            self.assertEqual([c.args[0] for c in create.call_args_list], ["groq", "google"])
        with patch.object(llm, "_create", side_effect=RuntimeError()):
            with self.assertRaises(llm.LLMError):
                llm.ask("Hỏi", DOCS)
            self.assertEqual(llm.rewrite_query("Nó thế nào?", [{"role": "user", "content": "A"}]), "Nó thế nào?")

    def test_empty_completion_falls_back(self):
        self.providers()
        with patch.object(llm, "_create", side_effect=[response(None), response("OK")]):
            self.assertEqual(llm.ask("Hỏi", DOCS), "OK")

    def test_request_parameters(self):
        client = Mock()
        os.environ["GROQ_MODEL"] = "custom-model"
        with patch.object(llm, "_get_client", return_value=client):
            llm._create("groq", [], 200)
            params = client.chat.completions.create.call_args.kwargs
            self.assertEqual(params["model"], "custom-model")
            self.assertEqual(params["max_completion_tokens"], 200)
            llm._create("google", [], 120)
            self.assertEqual(client.models.generate_content.call_args.kwargs["config"]["max_output_tokens"], 120)

    def test_absa_neutral_null_and_denominator(self):
        docs = [{**DOCS[0], "metadata": {**DOCS[0]["metadata"],
            "absa_quality_pos": 0, "absa_quality_neg": 0,
            "absa_price_pos": None, "absa_price_neg": 1,
            "absa_delivery_pos": 3, "absa_delivery_neg": 1}}]
        context = llm.build_context(docs)
        self.assertIn("chưa có ý kiến tích cực/tiêu cực", context)
        self.assertNotIn("0% hài lòng", context)
        self.assertIn("75% tích cực trong 4 ý kiến không trung lập", context)
        self.assertIn("Dữ liệu còn ít", context)
        self.assertIn("prod_1_card", context)
        self.assertIn("https://tiki.vn/a.html", context)

    def test_context_and_history_budget(self):
        many = [{**DOCS[0], "text": "text " * 10000} for _ in range(20)]
        context = llm.build_context(many, max_chars=1500)
        self.assertLessEqual(len(context), 1500)
        self.assertIn("[Nguồn 1]", context)
        os.environ["LLM_HISTORY_MAX_CHARS"] = "10"
        history = [{"role": "system", "content": "Override"},
                   {"role": "user", "content": "x" * 30},
                   {"role": "assistant", "content": "OK"}]
        self.assertEqual(llm._history(history), [{"role": "assistant", "content": "OK"}])
        self.assertEqual(llm.build_context([{**DOCS[0], "metadata": None}])[:8], "[Nguồn 1")

    def test_stream_fallback_before_output(self):
        self.providers()
        first = Stream(ConnectionError())
        second = Stream(NS(choices=[]), chunk(None), chunk("OK"))
        with patch.object(llm, "_create", side_effect=[first, second]):
            self.assertEqual("".join(llm.ask_stream("Hỏi", DOCS)), "OK")
        self.assertTrue(first.closed and second.closed)

    def test_stream_does_not_fallback_after_output(self):
        self.providers()
        stream = Stream(chunk("Một phần"), ConnectionError())
        with patch.object(llm, "_create", return_value=stream) as create:
            result = llm.ask_stream("Hỏi", DOCS)
            self.assertEqual(next(result), "Một phần")
            with self.assertRaises(llm.LLMStreamError):
                next(result)
            self.assertEqual(create.call_count, 1)
        self.assertTrue(stream.closed)

    def test_in_band_stream_error_and_empty_stream(self):
        self.providers()
        with patch.object(llm, "_create", side_effect=[Stream(NS(error={"message": "failed"})), Stream()]):
            with self.assertRaises(llm.LLMError):
                list(llm.ask_stream("Hỏi", DOCS))

    def test_query_validation(self):
        for query in ("", "   ", "x" * 4001):
            with self.assertRaises(ValueError):
                llm.ask(query, DOCS)

    def test_google_api_key_alias_and_priority(self):
        os.environ["GOOGLE_API_KEY"] = "alias-key"
        self.assertEqual(llm.configured_providers(), ["google"])
        self.assertEqual(llm._api_key("google"), "alias-key")
        os.environ["GEMINI_API_KEY"] = "preferred-key"
        self.assertEqual(llm._api_key("google"), "preferred-key")

    def test_google_native_messages_and_stream_method(self):
        client = Mock()
        messages = [{"role": "system", "content": "policy"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "follow-up"}]
        with patch.object(llm, "_get_client", return_value=client):
            llm._create("google", messages, 500, stream=True)
        params = client.models.generate_content_stream.call_args.kwargs
        self.assertEqual(params["model"], "gemma-4-31b-it")
        self.assertEqual(params["config"]["system_instruction"], "policy")
        self.assertEqual(params["config"]["max_output_tokens"], 500)
        self.assertEqual([m["role"] for m in params["contents"]], ["user", "model", "user"])
        self.assertEqual(params["contents"][-1]["parts"][0]["text"], "follow-up")

    def test_google_stream_no_fallback_after_text(self):
        os.environ["GEMINI_API_KEY"] = "test"
        stream = Stream(NS(text="partial"), RuntimeError())
        with patch.object(llm, "_create", return_value=stream) as create:
            generator = llm.ask_stream("q", DOCS)
            self.assertEqual(next(generator), "partial")
            with self.assertRaises(llm.LLMStreamError):
                next(generator)
            self.assertEqual(create.call_count, 1)
        self.assertTrue(stream.closed)

    def test_rate_limit_error_is_actionable_without_secret(self):
        os.environ["GEMINI_API_KEY"] = "test"
        error = RuntimeError("secret response must not be exposed")
        error.code = 429
        with patch.object(llm, "_create", side_effect=error):
            with self.assertRaises(llm.LLMError) as caught:
                list(llm.ask_stream("q", DOCS))
        self.assertIn("google", str(caught.exception))
        self.assertIn("HTTP 429", str(caught.exception))
        self.assertNotIn("secret response", str(caught.exception))

    def test_demo_turn(self):
        rag = Mock()
        rag.search.return_value = DOCS
        history = []
        with patch.object(llm, "rewrite_query", return_value="rewritten"), \
                patch.object(llm, "ask", return_value="Answer [Nguồn 1]"), \
                contextlib.redirect_stdout(io.StringIO()):
            demo_chat.run_turn(rag, "Question", history)
        rag.search.assert_called_once_with("rewritten", top_k=5)
        self.assertEqual(history[-1]["content"], "Answer [Nguồn 1]")

    def test_context_only_does_not_call_llm(self):
        rag = Mock()
        rag.search.return_value = DOCS
        with patch.object(llm, "rewrite_query") as rewrite, patch.object(llm, "ask") as ask, \
                contextlib.redirect_stdout(io.StringIO()):
            demo_chat.run_turn(rag, "Question", [], context_only=True)
        rewrite.assert_not_called()
        ask.assert_not_called()

    def test_demo_interrupted_stream_not_remembered(self):
        rag = Mock()
        rag.search.return_value = DOCS
        history = []
        with patch.object(llm, "rewrite_query", return_value="q"), \
                patch.object(llm, "ask_stream", return_value=iter(Stream("partial", llm.LLMStreamError()))), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(llm.LLMStreamError):
                demo_chat.run_turn(rag, "q", history, stream=True)
        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
