"""Run from project root: python -m chainlit run source/chatbot/demo_chainlit.py.

URL analysis can run without Google; normal chat needs a configured LLM provider.
The shared retrieval model is initialized on the first search, never at import.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
from pathlib import Path
import re
import sys
from threading import Lock

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
# Chainlit loads the target by file path, which may not add the project root.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env", override=False)

import chainlit as cl
from chainlit.input_widget import Select
from starlette.concurrency import iterate_in_threadpool
from runtime.chatbot import llm
from runtime.chatbot.retrieval import TikiRAG
from runtime.chatbot.url_analyzer import (
    ABSA_MODELS, AnalysisError, analyze_product, analysis_to_document,
    extract_product_id, format_analysis_report,
)

logger = logging.getLogger(__name__)
TIKI_URL_PATTERN = re.compile(r"https?://(?:www\.)?tiki\.vn/[^\s<>\"']+", re.IGNORECASE)
FOLLOWUP_HINTS = ("sản phẩm này", "sp này", "món này", "cái này", "hàng này", "nó", "link vừa gửi", "vừa gửi")
ALTERNATIVE_HINTS = ("thay thế", "sản phẩm khác", "sp khác", "tương tự", "cùng phân khúc", "so sánh với")
MAX_HISTORY = 8
_rag = None
_retrieval_lock = Lock()
_llm_lock = Lock()


def extract_tiki_url(text):
    match = TIKI_URL_PATTERN.search(text)
    return match.group(0).rstrip(".,;:!?)]}，。") if match else None


def detect_intent(query):
    query = query.lower()
    if any(hint in query for hint in ALTERNATIVE_HINTS):
        return "alternative_recommendation"
    if any(hint in query for hint in ("có nên mua", "nên mua không", "đáng mua", "đáng tiền", "ổn không")):
        return "purchase_advice"
    if any(hint in query for hint in ("review", "đánh giá", "ưu điểm", "nhược điểm", "phàn nàn")):
        return "review_summary"
    return "general_qa"


def is_followup_query(query):
    # Word boundaries avoid treating a substring in a different word as "nó".
    return any(re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", query.lower()) for hint in FOLLOWUP_HINTS)


def build_search_query(query, current_product, history):
    if current_product and detect_intent(query) == "alternative_recommendation":
        metadata = current_product["document"]["metadata"]
        return (f"{query}. Thay thế cho {metadata.get('name', '')}. "
                f"Danh mục: {metadata.get('category_name', '')}. Giá tham chiếu: {metadata.get('price', '')}.")[:4000]
    with _llm_lock:
        return llm.rewrite_query(query, history)


def _search(query, top_k=5):
    global _rag
    with _retrieval_lock:
        if _rag is None:
            _rag = TikiRAG()
        return _rag.search(query, top_k=top_k)


def _close_resources():
    global _rag
    if _rag is not None:
        _rag.close()
        _rag = None
    llm.close_clients()


atexit.register(_close_resources)


def _locked_stream(query, docs, history):
    with _llm_lock:
        yield from llm.ask_stream(query, docs, history)


async def stream_answer(query, docs, history):
    """Advance the synchronous SDK iterator in a worker, not on the UI loop."""
    generator = _locked_stream(query, docs, history)
    try:
        async for text in iterate_in_threadpool(generator):
            yield text
    finally:
        await asyncio.to_thread(generator.close)


def push_history(query, answer):
    history = list(cl.user_session.get("history") or [])
    history.extend([{"role": "user", "content": query[:1200]},
                    {"role": "assistant", "content": answer[:2500]}])
    cl.user_session.set("history", history[-MAX_HISTORY:])


async def _show_sources(docs):
    lines = []
    for index, doc in enumerate(docs, 1):
        metadata = doc.get("metadata") or {}
        label = str(metadata.get("name") or doc["id"])
        url = metadata.get("product_url")
        lines.append(f"[{index}] {label} — `{doc['id']}`" + (f"\n{url}" if url else ""))
    if lines:
        await cl.Message(content="### Nguồn tham khảo\n\n" + "\n\n".join(lines)).send()


@cl.set_starters
async def starters():
    return [cl.Starter(label="Tìm sản phẩm", message="Gợi ý sản phẩm chất lượng tốt"),
            cl.Starter(label="Phân tích URL", message="https://tiki.vn/product-p53497035.html")]


@cl.on_chat_start
async def start():
    await cl.ChatSettings([Select(id="absa_model", label="Model ABSA cho sản phẩm crawl mới",
        values=list(ABSA_MODELS), initial_value="logreg",
        description="Sản phẩm trong KB dùng thống kê có sẵn; model chọn ở đây chỉ dùng khi crawl mới.")]).send()
    cl.user_session.set("absa_model", "logreg")
    cl.user_session.set("history", [])
    cl.user_session.set("last_docs", [])
    cl.user_session.set("current_product", None)
    cl.user_session.set("busy", False)
    await cl.Message(content="Chào bạn! Hãy hỏi về sản phẩm hoặc gửi URL Tiki. "
        "Phân tích URL không cần key Google; nhận xét và chat cần key. "
        "Dùng /clear để xóa lịch sử và sản phẩm hiện tại.").send()


@cl.on_settings_update
async def update_settings(settings):
    model = settings.get("absa_model", "logreg")
    if model not in ABSA_MODELS:
        await cl.Message(content="Model ABSA không hợp lệ.").send()
        return
    cl.user_session.set("absa_model", model)


@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    if not query:
        return
    if cl.user_session.get("busy"):
        await cl.Message(content="Mình đang xử lý câu trước, bạn chờ hoàn tất nhé.").send()
        return
    if len(query) > 4000:
        await cl.Message(content="Câu hỏi tối đa 4.000 ký tự.").send()
        return
    if query.lower() == "/clear":
        for key, value in (("history", []), ("last_docs", []), ("current_product", None)):
            cl.user_session.set(key, value)
        await cl.Message(content="Đã xóa lịch sử và sản phẩm hiện tại.").send()
        return
    cl.user_session.set("busy", True)
    try:
        url = extract_tiki_url(query)
        if url or query.isascii() and query.isdigit():
            await handle_url_analysis(url or query, query)
        else:
            await handle_chat(query)
    finally:
        cl.user_session.set("busy", False)


async def handle_url_analysis(url, query=""):
    msg = await cl.Message(content="Đang phân tích URL...").send()
    cl.user_session.set("current_product", None)
    async def progress(text):
        msg.content = text
        await msg.update()
    try:
        analysis = await analyze_product(url, progress_callback=progress,
                                         absa_model=cl.user_session.get("absa_model", "logreg"))
        doc = analysis_to_document(analysis)
        cl.user_session.set("current_product", {"id": analysis.product_id, "document": doc})
        cl.user_session.set("last_docs", [doc])
        report = format_analysis_report(analysis)
        msg.content = report
        await msg.update()
        push_history(query or url, report)
        # URL alone is a deterministic report: do not spend LLM quota.
        question = query.replace(url, "").strip(" \n:.,?!()")
        if question and question.lower() not in ("phân tích", "phân tích sản phẩm", "phân tích url"):
            await handle_chat(question, focus_current=True)
    except (AnalysisError, ValueError) as exc:
        msg.content = f"Không phân tích được sản phẩm: {exc}"
        await msg.update()
    except Exception as exc:
        logger.warning("URL analysis failed (%s)", type(exc).__name__)
        msg.content = "Phân tích URL thất bại. Kiểm tra kết nối Tiki/Qdrant và model ABSA."
        await msg.update()


async def handle_chat(query, *, focus_current=False):
    msg = await cl.Message(content="Đang tìm thông tin...").send()
    full_answer = ""
    try:
        history = cl.user_session.get("history") or []
        current = cl.user_session.get("current_product")
        intent = detect_intent(query)
        generic_followup = query.lower().strip(" .?!") in (
            "có nên mua", "có nên mua không", "đáng mua không", "ổn không", "review đi",
            "ưu điểm là gì", "nhược điểm là gì", "chất lượng thế nào", "giao hàng thì sao")
        if current and (focus_current or is_followup_query(query) or generic_followup) and intent != "alternative_recommendation":
            # Anchor URL follow-ups to real report data, not unrelated vector hits.
            docs = [current["document"]]
        else:
            if not current and is_followup_query(query) and not history:
                msg.content = "Bạn gửi tên hoặc URL sản phẩm trước để mình biết đang hỏi sản phẩm nào nhé."
                await msg.update()
                return
            search_query = await asyncio.to_thread(build_search_query, query, current, history)
            docs = await asyncio.to_thread(_search, search_query, 12 if current and intent == "alternative_recommendation" else 5)
            if current and intent == "alternative_recommendation":
                candidates = [doc for doc in docs if str((doc.get("metadata") or {}).get("product_id")) != current["id"]]
                docs = [current["document"], *candidates[:7]]
        cl.user_session.set("last_docs", docs)
        msg.content = ""
        await msg.update()
        async for chunk in stream_answer(query, docs, history):
            full_answer += chunk
            await msg.stream_token(chunk)
        await msg.update()
        push_history(query, full_answer)
        await _show_sources(docs)
    except llm.LLMError as exc:
        # Keep partial text visible, but never save it as a successful conversation turn.
        msg.content = (full_answer + "\n\n" if full_answer else "") + f"Chưa hoàn tất câu trả lời: {exc}"
        await msg.update()
    except Exception as exc:
        logger.warning("Chat failed (%s)", type(exc).__name__)
        msg.content = (full_answer + "\n\n" if full_answer else "") + "Không xử lý được câu hỏi; kiểm tra Qdrant, model và cấu hình LLM."
        await msg.update()
