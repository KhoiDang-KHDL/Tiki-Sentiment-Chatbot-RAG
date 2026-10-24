"""Exercise UI callbacks with fake messages/sessions; no browser or LLM requests."""
from types import SimpleNamespace as NS
import threading
import unittest
from unittest.mock import patch, Mock

from runtime.chatbot import demo_chainlit_local as ui, llm
from runtime.chatbot.url_analyzer import AnalysisError, ProductAnalysis, analysis_to_document


def product():
    return ProductAnalysis(product_id="123", name="Sản phẩm A", price=100,
        rating=4.5, review_count=20, brand="Brand", category="Category",
        url="https://tiki.vn/a-p123.html", aspect_scores={})


class Session:
    def __init__(self):
        self.values = {"history": [], "current_product": None, "absa_model": "logreg"}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class Message:
    sent = []

    def __init__(self, content=""):
        self.content = content

    async def send(self):
        self.sent.append(self)
        return self

    async def update(self):
        return self

    async def stream_token(self, token):
        self.content += token


async def answer(*args, **kwargs):
    yield "Answer [Nguồn 1]"


class UITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Session()
        Message.sent = []
        self.fake = patch.object(ui, "cl", NS(user_session=self.session, Message=Message))
        self.fake.start()
        self.addCleanup(self.fake.stop)

    async def test_url_only_saves_structured_product_without_llm(self):
        with patch.object(ui, "analyze_product", return_value=product()), patch.object(ui, "stream_answer") as stream:
            await ui.handle_url_analysis("https://tiki.vn/a-p123.html", "https://tiki.vn/a-p123.html")
        current = self.session.get("current_product")
        self.assertEqual(current["id"], "123")
        self.assertEqual(current["document"]["metadata"]["name"], "Sản phẩm A")
        stream.assert_not_called()
        self.assertIn("Sản phẩm A", Message.sent[0].content)

    async def test_failed_new_url_does_not_keep_previous_product(self):
        self.session.set("current_product", {"id": "old"})
        with patch.object(ui, "analyze_product", side_effect=AnalysisError("Tiki HTTP 403")):
            await ui.handle_url_analysis("https://tiki.vn/a-p123.html")
        self.assertIsNone(self.session.get("current_product"))
        self.assertIn("403", Message.sent[0].content)
        self.assertEqual(self.session.get("history"), [])

    async def test_url_followup_uses_report_not_unrelated_search(self):
        doc = analysis_to_document(product())
        self.session.set("current_product", {"id": "123", "document": doc})
        with patch.object(ui, "_search") as search, patch.object(ui, "stream_answer", side_effect=answer) as stream:
            await ui.handle_chat("Sản phẩm này có tốt không?")
        search.assert_not_called()
        self.assertEqual(stream.call_args.args[1], [doc])
        self.assertEqual(self.session.get("history")[-1]["content"], "Answer [Nguồn 1]")

    async def test_alternative_excludes_same_product_but_keeps_context(self):
        doc = analysis_to_document(product())
        other = {**doc, "id": "other", "metadata": {"product_id": 456, "name": "B"}}
        self.session.set("current_product", {"id": "123", "document": doc})
        with patch.object(ui, "_search", return_value=[doc, other]), patch.object(ui, "stream_answer", side_effect=answer) as stream:
            await ui.handle_chat("Có sản phẩm thay thế không?")
        self.assertEqual([d["id"] for d in stream.call_args.args[1]], [doc["id"], "other"])

    async def test_named_new_question_does_not_reuse_old_url(self):
        self.session.set("current_product", {"id": "123", "document": analysis_to_document(product())})
        with patch.object(ui, "build_search_query", return_value="review Sony"), \
                patch.object(ui, "_search", return_value=[]) as search, patch.object(ui, "stream_answer", side_effect=answer):
            await ui.handle_chat("Review tai nghe Sony")
        search.assert_called_once()

    async def test_partial_stream_preserved_but_not_saved_in_history(self):
        async def broken(*args, **kwargs):
            yield "partial"
            raise llm.LLMStreamError("interrupted")
        with patch.object(ui, "build_search_query", return_value="q"), patch.object(ui, "_search", return_value=[]), \
                patch.object(ui, "stream_answer", side_effect=broken):
            await ui.handle_chat("q")
        self.assertIn("partial", Message.sent[0].content)
        self.assertIn("interrupted", Message.sent[0].content)
        self.assertEqual(self.session.get("history"), [])

    async def test_sdk_iterator_runs_outside_ui_thread(self):
        main_thread = threading.get_ident()
        def tokens(*args):
            self.assertNotEqual(threading.get_ident(), main_thread)
            yield "A"
            yield "B"
        with patch.object(llm, "ask_stream", side_effect=tokens):
            parts = [text async for text in ui.stream_answer("q", [], [])]
        self.assertEqual(parts, ["A", "B"])

    def test_history_is_bounded(self):
        for i in range(10):
            ui.push_history(str(i), "answer")
        self.assertEqual(len(self.session.get("history")), 8)

    def test_url_and_followup_extraction(self):
        self.assertEqual(ui.extract_tiki_url("Xem (https://tiki.vn/a-p123.html)."), "https://tiki.vn/a-p123.html")
        self.assertIsNone(ui.extract_tiki_url("https://tiki.vn.evil.vn/a-p123.html"))
        self.assertFalse(ui.is_followup_query("Cốc nóng lạnh"))
        self.assertTrue(ui.is_followup_query("nó có tốt không?"))


if __name__ == "__main__":
    unittest.main()
