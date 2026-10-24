from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
URL Analyzer — paste Tiki URL → phân tích sản phẩm real-time
3-tier: Redis cache → Qdrant KB → Crawl + ABSA + Index
"""
import re
import os
import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
import json
try:
    import redis as redis_client
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

logger = logging.getLogger("url_analyzer")

from src.crawling.module2_product_detail import fetch_product_detail
from src.crawling.module3_product_reviews import fetch_reviews_page
from src.absa.inference import predict_and_aggregate

ASPECTS    = ["description", "quality", "packaging", "delivery", "service", "price"]
ASPECTS_VI = {
    "description": "Mô tả SP",
    "quality":     "Chất lượng",
    "packaging":   "Đóng gói",
    "delivery":    "Giao hàng",
    "service":     "Dịch vụ",
    "price":       "Giá cả",
}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def _get_redis():
    try:
        r = redis_client.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
            socket_connect_timeout=2,
        )
        r.ping()
        return r
    except Exception:
        return None

from redis import Redis
_redis: Optional[Redis[str]] = _get_redis() if REDIS_AVAILABLE else None

@dataclass
class ProductAnalysis:
    product_id:     str
    name:           str
    price:          int
    rating:         float
    review_count:   int
    brand:          str
    category:       str
    url:            str
    aspect_scores:  dict
    sample_reviews: list[dict] = field(default_factory=list)
    source:         str = "kb"
    total_analyzed: int = 0
    absa_model_used: str = "logreg"


def extract_product_id(url: str) -> Optional[str]:
    match = re.search(r'-p(\d+)', url)
    if match:
        return match.group(1)
    match = re.search(r'product[_/]?(\d+)', url)
    if match:
        return match.group(1)
    if url.strip().isdigit():
        return url.strip()
    return None


def _check_qdrant_kb(product_id: str) -> ProductAnalysis | None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

        results = client.scroll(
            collection_name="tiki_kb",
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="metadata.product_id", match=qm.MatchValue(value=int(product_id))),
                qm.FieldCondition(key="doc_type",            match=qm.MatchValue(value="product_card")),
            ]),
            limit=1,
            with_payload=True,
        )[0]

        if not results:
            return None

        payload = results[0].payload or {}
        if not isinstance(payload, dict):
            payload = {}
        m = payload.get("metadata", {})

        aspect_scores  = {}
        total_analyzed = 0
        for asp in ASPECTS:
            pos   = m.get(f"absa_{asp}_pos", 0)
            neg   = m.get(f"absa_{asp}_neg", 0)
            score = m.get(f"absa_{asp}_score", 0)
            total = pos + neg
            pct   = round(pos / total * 100) if total > 0 else 0
            aspect_scores[asp] = {"score": score, "pos": pos, "neg": neg, "pct": pct, "total": total}
            total_analyzed = max(total_analyzed, total)

        review_results = client.scroll(
            collection_name="tiki_kb",
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="metadata.product_id", match=qm.MatchValue(value=int(product_id))),
                qm.FieldCondition(key="doc_type",            match=qm.MatchValue(value="review")),
            ]),
            limit=5,
            with_payload=True,
        )[0]

        sample_reviews = [
            {
                "rating":  (r.payload or {}).get("metadata", {}).get("review_rating", ""),
                "content": (r.payload or {}).get("content", "")[:200],
            }
            for r in review_results
        ]

        return ProductAnalysis(
            product_id=product_id,
            name=m.get("name", ""),
            price=m.get("price", 0),
            rating=m.get("rating_average", 0),
            review_count=m.get("review_count", 0),
            brand=m.get("brand_name", ""),
            category=m.get("category_name", ""),
            url=m.get("product_url", ""),
            aspect_scores=aspect_scores,
            sample_reviews=sample_reviews,
            source="kb",
            total_analyzed=total_analyzed,
            absa_model_used="kb_precomputed",
        )
    except Exception as e:
        logger.warning(f"Qdrant KB check failed: {e}")
        return None


def crawl_reviews_fast(product_id: str, max_pages: int = 5) -> List[dict]:
    collected = []
    for page in range(1, max_pages + 1):
        reviews, total_pages = fetch_reviews_page(product_id, page)
        if not reviews:
            break
        for rev in reviews:
            content = (rev.get("content") or "").strip()
            if content:
                collected.append({
                    "product_id":    product_id,
                    "customer_name": (rev.get("created_by") or {}).get("name", ""),
                    "rating":        rev.get("rating", ""),
                    "content":       content,
                })
        if page >= total_pages:
            break
    return collected


async def _crawl_and_analyze_async(
    product_id: str,
    progress_callback=None,
    absa_model: str = "logreg",
) -> ProductAnalysis | None:
    loop = asyncio.get_event_loop()

    if progress_callback:
        await progress_callback("🌐 Đang crawl sản phẩm + reviews song song...")

    detail_future  = loop.run_in_executor(executor, fetch_product_detail, product_id)
    reviews_future = loop.run_in_executor(executor, crawl_reviews_fast, product_id)

    raw, reviews = await asyncio.gather(detail_future, reviews_future)

    if not raw:
        return None

    if progress_callback:
        await progress_callback(f"✅ Crawl xong: {len(reviews)} reviews có nội dung")

    model_label = "⚡ LogReg" if absa_model == "logreg" else "🎯 PhoBERT"
    if progress_callback:
        await progress_callback(f"🤖 Đang phân tích ABSA ({model_label}) trên {len(reviews)} reviews...")

    aspect_scores = predict_and_aggregate(reviews, model=absa_model, version="v2")

    brand    = raw.get("brand")    or {}
    category = raw.get("categories") or {}

    sorted_reviews = sorted(reviews, key=lambda r: len(r.get("content", "")), reverse=True)
    sample_reviews = [
        {"rating": r["rating"], "content": r["content"][:200]}
        for r in sorted_reviews[:5]
    ]

    return ProductAnalysis(
        product_id=product_id,
        name=raw.get("name", ""),
        price=raw.get("price", 0),
        rating=raw.get("rating_average", 0),
        review_count=raw.get("review_count", 0),
        brand=brand.get("name", ""),
        category=category.get("name", ""),
        url=f"https://tiki.vn/{raw.get('url_path', '')}",
        aspect_scores=aspect_scores,
        sample_reviews=sample_reviews,
        source="crawl",
        total_analyzed=len(reviews),
        absa_model_used=absa_model,
    )


def format_analysis_report(analysis: ProductAnalysis) -> str:
    source_label = {
        "cache":          "⚡ Cache (< 0.1s)",
        "kb":             "📚 Knowledge Base (< 2s)",
        "crawl":          "🔄 Crawl + phân tích mới",
        "kb_precomputed": "📚 Knowledge Base (pre-computed)",
    }

    model_label = {
        "logreg":         "⚡ LogReg (F1=0.769)",
        "phobert":        "🎯 PhoBERT (F1=0.848)",
        "kb_precomputed": "📚 Pre-computed",
    }.get(analysis.absa_model_used, analysis.absa_model_used)

    lines = [
        "## 📊 Phân tích sản phẩm",
        "",
        f"**{analysis.name}**",
        f"💰 {analysis.price:,.0f}đ · ⭐ {analysis.rating}/5 ({analysis.review_count:,} đánh giá)",
        f"🏷️ {analysis.brand} · 📦 {analysis.category}",
        f"🔗 [Xem trên Tiki]({analysis.url})",
        "",
        f"*Nguồn: {source_label.get(analysis.source, analysis.source)} · ABSA: {model_label}*",
        "",
        f"### 🔍 Phân tích theo khía cạnh ({analysis.total_analyzed} đánh giá đã phân tích)",
        "",
    ]

    for asp in ASPECTS:
        data    = analysis.aspect_scores.get(asp, {})
        pct     = data.get("pct", 0)
        pos     = data.get("pos", 0)
        neg     = data.get("neg", 0)
        total   = data.get("total", 0)
        name_vi = ASPECTS_VI.get(asp, asp)

        if total == 0:
            emoji, note = "❓", "Chưa có đánh giá"
        elif pct >= 70:
            emoji, note = "✅", f"{pct}% hài lòng ({pos} khen / {neg} chê)"
        elif pct >= 40:
            emoji, note = "⚠️", f"{pct}% hài lòng ({pos} khen / {neg} chê)"
        else:
            emoji, note = "❌", f"Chỉ {pct}% hài lòng ({pos} khen / {neg} chê)"

        if 0 < total < 5:
            note += " · ⚠️ mẫu nhỏ"

        lines.append(f"{emoji} **{name_vi}**: {note}")

    if analysis.sample_reviews:
        lines += ["", "### 💬 Một số đánh giá tiêu biểu", ""]
        for rev in analysis.sample_reviews[:3]:
            rating  = rev.get("rating", "")
            content = rev.get("content", "")
            stars   = "⭐" * int(float(rating)) if rating else ""
            lines.append(f"> {stars} *\"{content}\"*")
            lines.append("")

    lines += ["---", f"*Xem đầy đủ đánh giá trên [Tiki]({analysis.url})*"]
    return "\n".join(lines)


async def analyze_url(
    url: str,
    progress_callback=None,
    absa_model: str = "logreg",
    user_query: str = "",
) -> str:
    product_id = extract_product_id(url)
    if not product_id:
        return "❌ Không thể nhận diện product ID từ URL. Vui lòng paste link Tiki hợp lệ."

    if progress_callback:
        await progress_callback(f"🔍 Đã nhận diện sản phẩm ID: {product_id}")

    # Tier 1: Redis cache
    if _redis:
        try:
            cached = _redis.get(f"analysis:{product_id}")
            if cached:
                if progress_callback:
                    await progress_callback("⚡ Lấy từ cache!")
                # Vẫn generate LLM recommendation nếu có user_query
                if user_query and user_query.strip() != url.strip():
                    from src.chatbot.llm import ask_url_recommendation
                    if progress_callback:
                        await progress_callback("💬 Đang tổng hợp nhận xét...")
                    recommendation = ask_url_recommendation(user_query, cached)
                    return cached + "\n\n---\n\n## 💡 Nhận xét\n\n" + recommendation
                return cached
        except Exception:
            pass

    # Tier 2: Qdrant KB
    if progress_callback:
        await progress_callback("📚 Kiểm tra Knowledge Base...")

    analysis = _check_qdrant_kb(product_id)

    if analysis:
        if progress_callback:
            await progress_callback(f"⚡ Tìm thấy trong KB — {analysis.name}")
    else:
        # Tier 3: Crawl + ABSA
        analysis = await _crawl_and_analyze_async(product_id, progress_callback, absa_model=absa_model)

        if not analysis:
            return "❌ Không tìm thấy sản phẩm trên Tiki. Vui lòng kiểm tra lại URL."

        if progress_callback:
            await progress_callback("✅ Phân tích hoàn tất!")

    report = format_analysis_report(analysis)

    # Lưu vào Redis cache (TTL 1 giờ)
    if _redis:
        try:
            _redis.set(f"analysis:{product_id}", report, ex=3600)
        except Exception:
            pass

    # LLM recommendation nếu user có câu hỏi
    if user_query and user_query.strip() != url.strip():
        from src.chatbot.llm import ask_url_recommendation
        if progress_callback:
            await progress_callback("💬 Đang tổng hợp nhận xét...")
        recommendation = ask_url_recommendation(user_query, report)
        report = report + "\n\n---\n\n## 💡 Nhận xét\n\n" + recommendation

    return report
