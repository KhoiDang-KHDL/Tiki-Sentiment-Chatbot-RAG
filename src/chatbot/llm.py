from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
LLM caller — Multi-provider (Groq → OpenRouter fallback)
Streaming + Optimized prompting for ABSA-aware RAG
"""
import os
import logging
from groq import Groq

logger = logging.getLogger("llm")

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# OpenRouter client (fallback)
_openrouter_client = None

def _get_openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )
    return _openrouter_client


ASPECTS_VI = {
    "description": "Mô tả SP",
    "quality":     "Chất lượng",
    "packaging":   "Đóng gói",
    "delivery":    "Giao hàng",
    "service":     "Dịch vụ",
    "price":       "Giá cả",
}

SYSTEM_PROMPT = """Bạn là Tiki Shopping Assistant — trợ lý mua sắm thông minh.

## Nguyên tắc
- Trả lời tự nhiên như đang trò chuyện, KHÔNG liệt kê khô khan
- Dựa 100% vào data được cung cấp, KHÔNG bịa
- Nếu không có data → nói thẳng "mình chưa có thông tin về sản phẩm này"

## Ghi nhớ hội thoại multi-turn
- Nếu trong lịch sử có [CURRENT_PRODUCT_CONTEXT], đó là sản phẩm hiện tại người dùng đang hỏi.
- Các cụm như "sản phẩm này", "sp này", "món này", "cái này", "hàng này", "nó", "sản phẩm mới gửi", "sản phẩm vừa gửi", "link vừa gửi" đều ám chỉ CURRENT_PRODUCT_CONTEXT.
- Nếu người dùng hỏi tiếp sau khi vừa gửi URL, phải ưu tiên dùng CURRENT_PRODUCT_CONTEXT trước.
- Không được trả lời như một câu hỏi mới hoàn toàn nếu câu hỏi có dấu hiệu follow-up.

## Điều hướng intent
- Nếu người dùng hỏi "người dùng nói gì", "review", "đánh giá", "phàn nàn" → tóm tắt review của sản phẩm hiện tại.
- Nếu người dùng hỏi "có nên mua", "đáng mua không", "ổn không" → tư vấn mua/không mua dựa trên ưu điểm, nhược điểm và mức độ rủi ro.
- Nếu người dùng hỏi "thay thế", "sản phẩm khác", "tương tự", "cùng phân khúc", "gợi ý sản phẩm khác" → chuyển sang chế độ gợi ý sản phẩm thay thế, không chỉ tóm tắt lại review sản phẩm hiện tại.
- Nếu dữ liệu retrieval không có sản phẩm thay thế phù hợp, hãy nói rõ rằng hiện tại chưa đủ dữ liệu để gợi ý chắc chắn, thay vì bịa tên sản phẩm.

## Cách diễn đạt số liệu ABSA
KHÔNG BAO GIỜ viết raw numbers như "score 0.87" hay "94 pos / 47 neg".
Hãy diễn đạt tự nhiên:

- Nhiều khen (>70%): "được đa số khách hàng đánh giá tích cực"
- Trái chiều (40-70%): "nhận được đánh giá trái chiều"
- Nhiều chê (<40%): "nhiều khách hàng phàn nàn"
- Ít đánh giá (< 10): "chưa có nhiều đánh giá để kết luận"

Khi cần cụ thể, dùng phần trăm tự nhiên:
✅ "Khoảng 70% khách hài lòng về giao hàng"
✅ "Chất lượng là điểm yếu — chỉ 1/3 khách hài lòng"
❌ "absa_quality_score: 0.333, pos: 94, neg: 47"

## Cấu trúc trả lời
- Mở đầu: trả lời thẳng vào câu hỏi (1-2 câu)
- Thân: phân tích với dẫn chứng tự nhiên
- Kết: khuyến nghị ngắn gọn
- Tối đa 250 từ
- Highlight cả ưu VÀ nhược điểm

## Khi gợi ý sản phẩm
- Nêu tên + giá + lý do cụ thể
- Gợi ý ai nên/không nên mua

## Khi so sánh
- Tóm tắt khác biệt chính
- Kết luận: "nếu cần X thì chọn A, nếu cần Y thì chọn B"

## Ngôn ngữ
- Tiếng Việt tự nhiên, thân thiện
- Dùng "mình/bạn"
- Emoji nhẹ nhàng, không quá nhiều"""


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.get("metadata", {})
        text = doc.get("text", "")
        doc_type = doc.get("doc_type", "")

        header = f"[Nguồn {i}] ({doc_type})"
        if m.get("name"):
            header += f" {m['name']}"
            if m.get("price"):
                header += f" — {m['price']:,.0f}đ"
            if m.get("rating_average"):
                header += f" — {m['rating_average']}/5 ({m.get('review_count', '?')} đánh giá)"

        absa_lines = []
        for asp_en, asp_vi in ASPECTS_VI.items():
            score = m.get(f"absa_{asp_en}_score")
            pos = m.get(f"absa_{asp_en}_pos", 0)
            neg = m.get(f"absa_{asp_en}_neg", 0)
            if score is not None:
                total = pos + neg
                pct = round(pos / total * 100) if total > 0 else 0
                absa_lines.append(f"  {asp_vi}: {pct}% hài lòng ({pos} khen / {neg} chê)")

        absa_block = ""
        if absa_lines:
            absa_block = "\nPhân tích đánh giá:\n" + "\n".join(absa_lines)

        content = text if text else "(không có nội dung text)"
        parts.append(f"{header}\n{content}{absa_block}")

    return "\n\n" + "=" * 50 + "\n\n".join(parts)


def _call_openrouter(messages: list[dict], max_tokens: int = 1024, stream: bool = False):
    """Fallback call to OpenRouter."""
    or_client = _get_openrouter()
    return or_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        stream=stream,
    )


def rewrite_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    messages = [
        {
            "role": "system",
            "content": """
Bạn là bộ rewrite query cho chatbot mua sắm Tiki.

Nhiệm vụ:
Viết lại câu hỏi mới nhất của người dùng thành một câu tìm kiếm đầy đủ ngữ cảnh.

Quy tắc:
- Chỉ trả về câu tìm kiếm mới, không giải thích.
- Nếu câu hỏi đã đủ rõ ràng thì giữ nguyên.
- Nếu lịch sử có [CURRENT_PRODUCT_CONTEXT], hãy dùng nó để hiểu các cụm:
  "sản phẩm này", "sp này", "món này", "cái này", "hàng này", "nó",
  "sản phẩm mới gửi", "sản phẩm vừa gửi", "link vừa gửi".
- Nếu người dùng hỏi "thay thế", "tương tự", "sản phẩm khác", "cùng phân khúc",
  hãy rewrite thành query tìm sản phẩm thay thế/cùng phân khúc với sản phẩm hiện tại.
- Không bịa tên sản phẩm nếu lịch sử không có.
""".strip()
        },
    ]

    messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    try:
        resp = client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=messages,
            max_tokens=120,
            temperature=0,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception:
        try:
            resp = _call_openrouter(messages, max_tokens=120)
            rewritten = resp.choices[0].message.content.strip()
            return rewritten if rewritten else query
        except Exception:
            return query


def ask(query: str, docs: list[dict], history: list[dict] = None) -> str:
    context = build_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Thông tin sản phẩm:\n\n{context}\n\nCâu hỏi: {query}"})

    # Try Groq first, fallback OpenRouter
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq failed ({e}), falling back to OpenRouter")
        resp = _call_openrouter(messages)
        return resp.choices[0].message.content


def ask_stream(query: str, docs: list[dict], history: list[dict] = None):
    context = build_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Thông tin sản phẩm:\n\n{context}\n\nCâu hỏi: {query}"})

    # Try Groq first
    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
            stream=True,
        )
        first_chunk = True
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_chunk:
                    first_chunk = False
                yield delta if delta else ""
    except Exception as e:
        logger.warning(f"Groq stream failed ({e}), falling back to OpenRouter")
        stream = _call_openrouter(messages, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def ask_url_recommendation(query: str, report: str, history: list[dict] = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"Báo cáo phân tích sản phẩm:\n\n{report}\n\nCâu hỏi của người dùng: {query}"
    })

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq failed ({e}), falling back to OpenRouter")
        resp = _call_openrouter(messages, max_tokens=512)
        return resp.choices[0].message.content
