"""HTTP Chainlit client. All model execution runs in the API service."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
import chainlit as cl
from chainlit.input_widget import Select
from runtime.chatbot import api_client

URL_PATTERN = re.compile(r"https?://(?:www\.)?tiki\.vn/[^\s<>\"']+", re.IGNORECASE)
ABSA_MODELS = ("logreg", "phobert_onnx", "phobert")


def push_history(query, answer):
    history = list(cl.user_session.get("history") or [])
    history.extend([{"role": "user", "content": query[:1200]},
                    {"role": "assistant", "content": answer[:2500]}])
    cl.user_session.set("history", history[-8:])


@cl.set_starters
async def starters():
    return [cl.Starter(label="Tìm sản phẩm", message="Gợi ý sản phẩm chất lượng tốt"),
            cl.Starter(label="Phân tích URL", message="https://tiki.vn/product-p53497035.html")]


@cl.on_chat_start
async def start():
    await cl.ChatSettings([Select(id="absa_model", label="Model ABSA khi crawl review mới",
                                values=list(ABSA_MODELS), initial_value="logreg")]).send()
    for key, value in (("history", []), ("context_id", None), ("busy", False), ("absa_model", "logreg")):
        cl.user_session.set(key, value)
    await cl.Message(content="Chào bạn! Hãy hỏi về sản phẩm hoặc gửi URL Tiki. Dùng /clear để bắt đầu lại.").send()


@cl.on_settings_update
async def update_settings(settings):
    model = settings.get("absa_model", "logreg")
    if model in ABSA_MODELS:
        cl.user_session.set("absa_model", model)


async def show_sources(sources):
    if not sources:
        return
    lines = []
    for index, source in enumerate(sources, 1):
        lines.append(f"[{index}] {source.get('name') or source['id']} — `{source['id']}`")
        if source.get("product_url"):
            lines.append(str(source["product_url"]))
    await cl.Message(content="### Nguồn tham khảo\n\n" + "\n\n".join(lines)).send()


async def handle_chat(query, *, focus_current=False):
    msg = await cl.Message(content="Đang tìm thông tin qua API...").send()
    answer, sources = "", []
    try:
        async for event, data in api_client.stream_chat(query, cl.user_session.get("history") or [],
                context_id=cl.user_session.get("context_id"), focus_current=focus_current):
            if event == "sources":
                sources = data
                msg.content = ""
                await msg.update()
            elif event == "token":
                text = data.get("text", "")
                answer += text
                await msg.stream_token(text)
        await msg.update()
        push_history(query, answer)
        await show_sources(sources)
    except api_client.APIError as exc:
        msg.content = (answer + "\n\n" if answer else "") + f"Chưa hoàn tất câu trả lời: {exc}"
        await msg.update()


async def handle_url_analysis(url, query=""):
    msg = await cl.Message(content="Đang phân tích sản phẩm qua API...").send()
    cl.user_session.set("context_id", None)
    try:
        result = await api_client.analyze(url, cl.user_session.get("absa_model", "logreg"))
        cl.user_session.set("context_id", result["context_id"])
        msg.content = result["report"]
        await msg.update()
        push_history(query or url, result["report"])
        question = query.replace(url, "").strip(" \n:.,?!()")
        if question and question.lower() not in ("phân tích", "phân tích sản phẩm", "phân tích url"):
            await handle_chat(question, focus_current=True)
    except api_client.APIError as exc:
        msg.content = f"Không phân tích được sản phẩm: {exc}"
        await msg.update()


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
        cl.user_session.set("history", [])
        cl.user_session.set("context_id", None)
        await cl.Message(content="Đã xóa lịch sử và sản phẩm hiện tại.").send()
        return
    cl.user_session.set("busy", True)
    try:
        match = URL_PATTERN.search(query)
        url = match.group(0).rstrip(".,;:!?)]}，。") if match else None
        if url or query.isascii() and query.isdigit():
            await handle_url_analysis(url or query, query)
        else:
            await handle_chat(query)
    finally:
        cl.user_session.set("busy", False)
