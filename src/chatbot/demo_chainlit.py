from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Chainlit UI — Tiki Shopping Assistant
Supports: normal chat + URL analysis (3-tier) + multi-turn memory
"""


import re

import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv

load_dotenv()

from src.chatbot.retrieval import TikiRAG
from src.chatbot.llm import ask_stream, rewrite_query
from src.chatbot.url_analyzer import analyze_url


rag = TikiRAG()

TIKI_URL_PATTERN = re.compile(r"(https?://(?:www\.)?tiki\.vn/\S+)")
PRODUCT_ID_PATTERN = re.compile(r"(?:-p|pid=)(\d+)")

MAX_HISTORY = 8
MAX_PRODUCT_CONTEXT_CHARS = 3500


FOLLOWUP_HINTS = [
    "sản phẩm này",
    "sp này",
    "món này",
    "cái này",
    "hàng này",
    "nó",
    "link vừa gửi",
    "link mới gửi",
    "sản phẩm vừa gửi",
    "sản phẩm mới gửi",
    "mới gửi",
    "vừa gửi",
]

ALTERNATIVE_STRONG_HINTS = [
    "thay thế",
    "sản phẩm khác",
    "sp khác",
    "mẫu khác",
    "món khác",
    "tương tự",
    "cùng phân khúc",
    "alternative",
    "so sánh với",
]

PURCHASE_HINTS = [
    "có nên mua",
    "nên mua không",
    "đáng mua",
    "đáng tiền",
    "ổn không",
    "ok không",
    "có tốt không",
]

REVIEW_HINTS = [
    "người dùng nói gì",
    "khách nói gì",
    "review",
    "đánh giá",
    "nhận xét",
    "phàn nàn",
    "ưu điểm",
    "nhược điểm",
    "vấn đề lớn nhất",
    "điểm yếu",
    "điểm mạnh",
]


def compact_text(text: str, max_chars: int = 2500) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[đã rút gọn]"


def extract_product_id(url: str) -> Optional[str]:
    match = PRODUCT_ID_PATTERN.search(url)
    return match.group(1) if match else None


def clean_markdown_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[#>\-\s]+", "", line)
    line = line.replace("**", "").replace("__", "")
    line = line.strip(" :-|")
    return line.strip()


def extract_product_name_from_report(report: str) -> Optional[str]:
    if not report:
        return None
    lines = [ln.strip() for ln in report.splitlines()]
    for i, line in enumerate(lines):
        if "phân tích sản phẩm" in line.lower():
            for candidate in lines[i + 1 : i + 8]:
                clean = clean_markdown_line(candidate)
                if not clean:
                    continue
                if clean.startswith("---"):
                    continue
                if "phân tích sản phẩm" in clean.lower():
                    continue
                if clean.startswith(("💰", "⭐", "🏷", "📦", "💡", "📊", "🔍")):
                    continue
                if len(clean) >= 8:
                    return clean
    bold_candidates = re.findall(r"\*\*(.+?)\*\*", report)
    for candidate in bold_candidates:
        clean = clean_markdown_line(candidate)
        if len(clean) >= 8 and "phân tích" not in clean.lower():
            return clean
    return None


def extract_price_text_from_report(report: str) -> Optional[str]:
    if not report:
        return None
    match = re.search(r"(\d{1,3}(?:[,.]\d{3})+)\s*đ", report)
    if match:
        return match.group(0)
    return None


def extract_category_from_report(report: str) -> Optional[str]:
    if not report:
        return None
    match = re.search(r"📦\s*([^\n]+)", report)
    if match:
        return clean_markdown_line(match.group(1))
    return None


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ALTERNATIVE_STRONG_HINTS):
        return "alternative_recommendation"
    if any(x in q for x in PURCHASE_HINTS):
        return "purchase_advice"
    if any(x in q for x in REVIEW_HINTS):
        return "review_summary"
    return "general_qa"


def is_followup_query(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in FOLLOWUP_HINTS)


def build_current_product_context(current_product: dict[str, Any] | None) -> str:
    if not current_product:
        return ""
    product_id = current_product.get("id") or "Không rõ ID"
    name = current_product.get("name") or "Không rõ tên"
    url = current_product.get("url") or ""
    price_text = current_product.get("price_text") or "Không rõ giá"
    category = current_product.get("category") or "Không rõ ngành hàng"
    report = compact_text(current_product.get("report", ""), MAX_PRODUCT_CONTEXT_CHARS)
    return f"""
[CURRENT_PRODUCT_CONTEXT]
Sản phẩm hiện tại trong hội thoại:
- Product ID: {product_id}
- Tên sản phẩm: {name}
- Giá: {price_text}
- Ngành hàng: {category}
- URL: {url}

Báo cáo phân tích gần nhất:
{report}
[/CURRENT_PRODUCT_CONTEXT]
""".strip()


def build_search_query(
    query: str,
    current_product: dict[str, Any] | None,
    history: list[dict[str, str]],
) -> str:
    intent = detect_intent(query)

    if not current_product:
        if history:
            return rewrite_query(query, history)
        return query

    product_name = current_product.get("name") or ""
    product_id = current_product.get("id") or ""
    price_text = current_product.get("price_text") or ""
    category = current_product.get("category") or ""

    if intent == "alternative_recommendation":
        return (
            f"{query}. Tìm sản phẩm thay thế hoặc sản phẩm tương tự cùng phân khúc "
            f"cho sản phẩm hiện tại: {product_name}. "
            f"Ngành hàng: {category}. Giá tham chiếu: {price_text}. Product ID: {product_id}."
        )

    if is_followup_query(query):
        return (
            f"{query}. Câu hỏi đang nói về sản phẩm hiện tại: "
            f"{product_name}. Ngành hàng: {category}. Giá: {price_text}. Product ID: {product_id}."
        )

    if history:
        return rewrite_query(query, history)
    return query


def build_augmented_history(
    history: list[dict[str, str]],
    current_product: dict[str, Any] | None,
    query: str,
) -> list[dict[str, str]]:
    intent = detect_intent(query)
    should_inject = (
        current_product is not None
        and (
            is_followup_query(query)
            or intent in ["alternative_recommendation", "purchase_advice", "review_summary"]
        )
    )
    if not should_inject:
        return history
    product_context = build_current_product_context(current_product)
    memory_message = {
        "role": "assistant",
        "content": (
            "Ghi nhớ ngữ cảnh hội thoại hiện tại. "
            "Các cụm như 'sản phẩm này', 'sp này', 'cái này', 'nó', "
            "'sản phẩm mới gửi', 'sản phẩm vừa gửi', 'link vừa gửi' "
            "đều ám chỉ sản phẩm dưới đây:\n\n"
            f"{product_context}"
        ),
    }
    return [memory_message] + history


def push_history(user_content: str, assistant_content: str):
    history = cl.user_session.get("history", []) or []
    history.append({
        "role": "user",
        "content": compact_text(user_content, 1200),
    })
    history.append({
        "role": "assistant",
        "content": compact_text(assistant_content, 2500),
    })
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    cl.user_session.set("history", history)


@cl.set_starters
async def starters(user: cl.User | None = None, message: Optional[str] = None):
    return [
        cl.Starter(
            label="🥤 Cốc giữ nhiệt tốt",
            message="Cốc giữ nhiệt chất lượng tốt, giữ nhiệt lâu",
        ),
        cl.Starter(
            label="🏠 Đồ gia dụng dưới 300k",
            message="Gợi ý đồ gia dụng phổ biến có giá dưới 300.000 VND",
        ),
        cl.Starter(
            label="🚚 Giao hàng nhanh",
            message="Sản phẩm nào được khách đánh giá giao hàng nhanh",
        ),
        cl.Starter(
            label="🔗 Phân tích URL",
            message=(
                "Phân tích sản phẩm: "
                "https://tiki.vn/binh-giu-nhiet-lock-lock-energetic-one-touch-tumbler-lhc3249-550ml-p83412126.html"
            ),
        ),
    ]


@cl.on_chat_start
async def start():
    await cl.ChatSettings([
        Select(
            id="absa_model",
            label="🤖 ABSA Model (khi phân tích URL mới)",
            values=["logreg", "phobert_onnx", "phobert"],
            initial_value="logreg",
            description=(
                "⚡ LogReg: nhanh ~2s | "
                "🚀 PhoBERT ONNX: nhanh ~5s | "
                "🎯 PhoBERT: chính xác ~44s"
            ),
        ),
    ]).send()

    cl.user_session.set("absa_model", "logreg")
    cl.user_session.set("history", [])
    cl.user_session.set("last_docs", [])
    cl.user_session.set("current_product", None)

    await cl.Message(
        content=(
            "👋 Xin chào! Mình là **Tiki Shopping Assistant**.\n\n"
            "Bạn có thể:\n"
            "- 💬 Hỏi về sản phẩm trên Tiki\n"
            "- 🔗 Paste link Tiki để phân tích chi tiết\n\n"
            "⚙️ Click icon cài đặt góc trên để chọn ABSA model!"
        ),
    ).send()


@cl.on_settings_update
async def update_settings(settings):
    model = settings.get("absa_model", "logreg")
    cl.user_session.set("absa_model", model)
    if model == "logreg":
        label = "LogReg nhanh"
    elif model == "phobert_onnx":
        label = "PhoBERT ONNX nhanh"
    else:
        label = "PhoBERT chính xác"
    await cl.Message(content=f"Đã chuyển sang **{label}**").send()


@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    url_match = TIKI_URL_PATTERN.search(query)
    if url_match:
        await handle_url_analysis(url_match.group(1), query=query)
    else:
        await handle_chat(query)


async def handle_url_analysis(url: str, query: str = ""):
    msg = cl.Message(content="")
    await msg.send()

    absa_model = cl.user_session.get("absa_model", "logreg")
    progress_lines: list[str] = []
    product_id = extract_product_id(url)

    cl.user_session.set("current_product", {
        "id": product_id,
        "url": url,
        "name": None,
        "price_text": None,
        "category": None,
        "report": "",
        "absa_model": absa_model,
        "status": "analyzing",
    })

    async def progress_callback(text: str):
        progress_lines.append(text)
        msg.content = "\n".join(progress_lines)
        await msg.update()

    try:
        report = await analyze_url(
            url,
            progress_callback=progress_callback,
            absa_model=absa_model,
            user_query=query,
        )

        if report is None:
            report = ""

        product_name = extract_product_name_from_report(report)
        price_text = extract_price_text_from_report(report)
        category = extract_category_from_report(report)

        current_product = {
            "id": product_id,
            "url": url,
            "name": product_name,
            "price_text": price_text,
            "category": category,
            "report": report,
            "absa_model": absa_model,
            "status": "done",
        }

        cl.user_session.set("current_product", current_product)

        history_assistant_content = (
            "Đã phân tích URL sản phẩm Tiki và lưu làm sản phẩm hiện tại trong hội thoại.\n\n"
            f"{build_current_product_context(current_product)}"
        )
        push_history(query or f"Phân tích sản phẩm: {url}", history_assistant_content)

        msg.content = "\n".join(progress_lines) + "\n\n---\n\n" + report
        await msg.update()

    except Exception as e:
        current_product = cl.user_session.get("current_product") or {}
        current_product["status"] = "error"
        current_product["error"] = f"{type(e).__name__}: {e}"
        cl.user_session.set("current_product", current_product)

        msg.content = (
            "\n".join(progress_lines)
            + "\n\n---\n\n"
            + f"❌ Lỗi khi phân tích URL: `{type(e).__name__}: {e}`\n\n"
            + "Bạn kiểm tra terminal Chainlit để xem traceback chi tiết."
        )
        await msg.update()


async def handle_chat(query: str):
    msg = cl.Message(content="🤔 Mình đang tìm kiếm...")
    await msg.send()

    try:
        history = cl.user_session.get("history", []) or []
        current_product = cl.user_session.get("current_product")

        if current_product and current_product.get("status") == "analyzing":
            msg.content = (
                "Mình vẫn đang phân tích sản phẩm vừa gửi. "
                "Bạn đợi report hiện ra xong rồi hỏi tiếp nhé."
            )
            await msg.update()
            return

        if current_product and current_product.get("status") == "error":
            msg.content = (
                "Lần phân tích URL trước bị lỗi nên mình chưa có đủ dữ liệu review để trả lời chắc chắn. "
                "Bạn thử gửi lại link hoặc kiểm tra log terminal giúp mình."
            )
            await msg.update()
            return

        if not current_product and is_followup_query(query):
            msg.content = (
                "Mình chưa có sản phẩm hiện tại trong hội thoại. "
                "Bạn gửi link Tiki trước để mình phân tích rồi hỏi tiếp nhé."
            )
            await msg.update()
            return

        intent = detect_intent(query)

        search_query = await cl.make_async(build_search_query)(
            query,
            current_product,
            history,
        )

        augmented_history = build_augmented_history(
            history,
            current_product,
            query,
        )

        top_k = 8 if intent == "alternative_recommendation" else 5

        async with cl.Step(name="🔍 Tìm kiếm", type="retrieval") as step:
            docs = await cl.make_async(rag.search)(search_query, top_k=top_k)
            cl.user_session.set("last_docs", docs)

            if search_query != query:
                step.output = f'Tìm: "{search_query}" → {len(docs)} sản phẩm'
            else:
                step.output = f"{len(docs)} sản phẩm"

        llm_query = query

        if current_product:
            if intent == "alternative_recommendation":
                llm_query = (
                    f"{query}\n\n"
                    "Yêu cầu trả lời:\n"
                    "- Người dùng đang hỏi gợi ý sản phẩm thay thế cho sản phẩm hiện tại.\n"
                    "- Không chỉ tóm tắt lại review của sản phẩm hiện tại.\n"
                    "- Nếu retrieval có sản phẩm phù hợp, hãy đề xuất và so sánh ngắn gọn.\n"
                    "- Nếu dữ liệu chưa đủ để gợi ý chắc chắn, hãy nói rõ hiện tại chưa đủ dữ liệu "
                    "và cần crawl/thêm sản phẩm tương tự."
                )
            elif intent == "purchase_advice":
                llm_query = (
                    f"{query}\n\n"
                    "Yêu cầu trả lời:\n"
                    "- Tư vấn có nên mua sản phẩm hiện tại không.\n"
                    "- Dựa trên review, điểm mạnh, điểm yếu, giá/rating nếu có.\n"
                    "- Kết luận rõ: nên mua nếu ai phù hợp, không nên mua nếu ai không phù hợp."
                )
            elif intent == "review_summary":
                llm_query = (
                    f"{query}\n\n"
                    "Yêu cầu trả lời:\n"
                    "- Tóm tắt người dùng đang đánh giá gì về sản phẩm hiện tại.\n"
                    "- Nêu cả điểm khen và điểm chê.\n"
                    "- Không bịa thêm thông tin ngoài dữ liệu."
                )

        msg.content = ""
        await msg.update()

        full_answer = ""
        for chunk in ask_stream(llm_query, docs, history=augmented_history):
            full_answer += chunk
            msg.content = full_answer
            await msg.update()

        push_history(query, full_answer)

        sources = []
        for d in docs:
            metadata = d.get("metadata", {}) or {}
            score = d.get("score", 0) or 0
            sources.append({
                "doc_type": d.get("doc_type", ""),
                "name": metadata.get("name", ""),
                "score": round(float(score), 3),
            })

        if sources:
            source_text = "\n".join([
                f"**{i + 1}.** {s['name']} · `{s['doc_type']}` · ⭐ {s['score']:.3f}"
                for i, s in enumerate(sources)
            ])
            await cl.Message(content=f"### 📚 Nguồn tham khảo\n\n{source_text}").send()

    except Exception as e:
        msg.content = f"❌ Lỗi: `{type(e).__name__}: {e}`"
        await msg.update()
