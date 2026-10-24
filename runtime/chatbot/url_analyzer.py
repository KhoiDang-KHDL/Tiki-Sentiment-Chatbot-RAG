"""Tiki URL analysis: optional Redis -> configured Qdrant -> live HTTP + ABSA.

Run: python -m runtime.chatbot.url_analyzer --url https://tiki.vn/product-p53497035.html
This module does not index newly crawled products or call an LLM unless requested.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import re
from threading import Lock
from urllib.parse import parse_qs, urlparse

from runtime.absa.inference import ASPECTS, predict_and_aggregate
from runtime.crawling.client import CrawlError
from runtime.crawling.module2_product_detail import fetch_product_detail
from runtime.crawling.module3_product_reviews import crawl_reviews_for_product
from runtime.preprocessing.clean_review import prepare_reviews

logger = logging.getLogger(__name__)
ABSA_MODELS = ("logreg", "phobert_onnx", "phobert")
ASPECTS_VI = dict(zip(ASPECTS, ("Đúng mô tả", "Chất lượng", "Đóng gói", "Giao hàng", "Dịch vụ", "Giá cả")))
_inference_lock = Lock()


# Keep the public exception name used by Chainlit and API callers.
AnalysisError = CrawlError


@dataclass
class ProductAnalysis:
    product_id: str
    name: str
    price: int | None
    rating: float | None
    review_count: int | None
    brand: str
    category: str
    url: str
    aspect_scores: dict
    sample_reviews: list[dict] = field(default_factory=list)
    source: str = "kb"
    total_analyzed: int = 0
    absa_model_used: str = "kb_precomputed"
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = field(default_factory=list)


def extract_product_id(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        candidate = value
    else:
        try:
            parsed = urlparse(value)
            if (parsed.scheme not in ("http", "https") or parsed.hostname not in ("tiki.vn", "www.tiki.vn")
                    or parsed.username or parsed.password or parsed.port not in (None, 80, 443)):
                return None
            match = re.search(r"-p([0-9]+)(?:\.html)?(?:/|$)", parsed.path)
            candidate = match.group(1) if match else parse_qs(parsed.query).get("product_id", [""])[0]
        except ValueError:
            return None
    if not re.fullmatch(r"[0-9]{1,19}", candidate) or not 0 < int(candidate) <= 2**63 - 1:
        return None
    return str(int(candidate))


def _number(value, integer=False):
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return None
        return int(number) if integer else number
    except (ValueError, TypeError, OverflowError):
        return None


def crawl_reviews_fast(product_id: str, max_pages: int = 5) -> list[dict]:
    """Live wrapper around the same crawler used by batch collection."""
    return prepare_reviews(crawl_reviews_for_product(product_id, max_pages=max_pages))


def _check_qdrant_kb(product_id: str) -> ProductAnalysis | None:
    from qdrant_client import QdrantClient, models as qm
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY"), timeout=5)
    collection = os.getenv("QDRANT_COLLECTION", "tiki_kb")
    def get(doc_type, limit):
        return client.scroll(collection_name=collection, scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="metadata.product_id", match=qm.MatchValue(value=int(product_id))),
            qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type)),
        ]), limit=limit, with_payload=True, with_vectors=False)[0]
    try:
        cards = get("product_card", 1)
        if not cards:
            return None
        metadata = (cards[0].payload or {}).get("metadata") or {}
        scores = {}
        for aspect in ASPECTS:
            pos = _number(metadata.get(f"absa_{aspect}_pos"), True) or 0
            neg = _number(metadata.get(f"absa_{aspect}_neg"), True) or 0
            total = pos + neg
            scores[aspect] = {"pos": pos, "neg": neg, "total": total,
                              "pct": round(100 * pos / total) if total else None,
                              "score": (pos - neg) / total if total else None}
        samples = []
        for point in get("review", 5):
            payload = point.payload or {}
            samples.append({"rating": (payload.get("metadata") or {}).get("review_rating"),
                            "content": str(payload.get("content") or "")[:600]})
        return ProductAnalysis(product_id=product_id, name=metadata.get("name") or f"Sản phẩm {product_id}",
            price=_number(metadata.get("price"), True), rating=_number(metadata.get("rating_average")),
            review_count=_number(metadata.get("review_count"), True), brand=metadata.get("brand_name") or "",
            category=metadata.get("category_name") or "", url=f"https://tiki.vn/product-p{product_id}.html",
            aspect_scores=scores, sample_reviews=samples,
            total_analyzed=_number(metadata.get("absa_n_reviews"), True) or 0,
            warnings=["Thông tin trong KB là dữ liệu đã thu thập, không phải giá/tồn kho thời gian thực."])
    finally:
        client.close()


def _cache(action, key, value=None):
    if not os.getenv("REDIS_URL"):
        return None
    try:
        import redis
        with redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                            socket_connect_timeout=1, socket_timeout=1) as client:
            if action == "get":
                cached = client.get(key)
                return json.loads(cached) if cached else None
            client.set(key, json.dumps(value, ensure_ascii=False, allow_nan=False), ex=3600)
    except Exception as exc:
        logger.warning("Analysis cache unavailable (%s)", type(exc).__name__)
    return None


async def _notify(callback, message):
    if callback:
        await callback(message)


def _infer(reviews, model):
    with _inference_lock:
        return predict_and_aggregate(reviews, model=model)


async def analyze_product(url: str, progress_callback=None, absa_model="logreg", *, force_refresh=False) -> ProductAnalysis:
    pid = extract_product_id(url)
    if not pid:
        raise ValueError("Vui lòng gửi URL tiki.vn hợp lệ có product ID hoặc một ID dạng số.")
    if absa_model not in ABSA_MODELS:
        raise ValueError("ABSA model must be logreg, phobert_onnx or phobert")
    await _notify(progress_callback, f"Đang phân tích sản phẩm {pid}...")
    namespace = hashlib.sha256(os.getenv("QDRANT_COLLECTION", "tiki_kb").encode()).hexdigest()[:12]
    key = f"analysis:v4:{namespace}:{absa_model}:{pid}"
    if not force_refresh:
        cached = await asyncio.to_thread(_cache, "get", key)
        if isinstance(cached, dict):
            try:
                analysis = ProductAnalysis(**cached)
                if analysis.product_id == pid:
                    analysis.source = "cache"
                    return analysis
            except (TypeError, ValueError):
                pass
        await _notify(progress_callback, "Kiểm tra Knowledge Base...")
        try:
            analysis = await asyncio.to_thread(_check_qdrant_kb, pid)
        except Exception as exc:
            logger.warning("KB lookup failed (%s)", type(exc).__name__)
            analysis = None
        if analysis is not None:
            await asyncio.to_thread(_cache, "set", key, asdict(analysis))
            return analysis

    await _notify(progress_callback, "Đang lấy thông tin sản phẩm và tối đa 5 trang đánh giá từ Tiki...")
    raw, reviews = await asyncio.gather(asyncio.to_thread(fetch_product_detail, pid),
                                        asyncio.to_thread(crawl_reviews_fast, pid), return_exceptions=True)
    if isinstance(raw, Exception):
        raise raw
    warnings = []
    if isinstance(reviews, Exception):
        warnings.append("Không lấy được đánh giá từ Tiki; chỉ có thông tin sản phẩm, chưa tính ABSA.")
        reviews = []
    if reviews:
        await _notify(progress_callback, f"Đang chạy {absa_model} trên {len(reviews)} đánh giá có nội dung...")
    try:
        scores = await asyncio.to_thread(_infer, reviews, absa_model)
    except Exception as exc:
        logger.warning("ABSA inference failed (%s)", type(exc).__name__)
        raise AnalysisError(f"Không chạy được model {absa_model}; kiểm tra checkpoint và thư viện trong .venv.") from exc
    brand, category = raw.get("brand") or {}, raw.get("categories") or {}
    if isinstance(category, list):
        category = category[-1] if category else {}
    analysis = ProductAnalysis(product_id=pid, name=raw.get("name") or f"Sản phẩm {pid}",
        price=_number(raw.get("price"), True), rating=_number(raw.get("rating_average")),
        review_count=_number(raw.get("review_count"), True),
        brand=brand.get("name", "") if isinstance(brand, dict) else "",
        category=category.get("name", "") if isinstance(category, dict) else "",
        url=f"https://tiki.vn/product-p{pid}.html", aspect_scores=scores,
        sample_reviews=sorted(reviews, key=lambda review: len(review["content"]), reverse=True)[:5],
        source="crawl", total_analyzed=len(reviews), absa_model_used=absa_model, warnings=warnings)
    if not warnings:
        await asyncio.to_thread(_cache, "set", key, asdict(analysis))
    return analysis


def format_analysis_report(analysis: ProductAnalysis) -> str:
    price = f"{analysis.price:,}đ" if analysis.price is not None else "Chưa có giá"
    rating = f"{analysis.rating}/5" if analysis.rating is not None else "Chưa có rating"
    lines = ["## Phân tích sản phẩm", f"**{analysis.name}**", f"{price} · {rating}",
             f"Thương hiệu: {analysis.brand or 'Chưa rõ'} · Danh mục: {analysis.category or 'Chưa rõ'}",
             f"[Xem trên Tiki]({analysis.url})", f"Nguồn: {analysis.source} · ABSA: {analysis.absa_model_used}",
             f"Số đánh giá đã phân tích: {analysis.total_analyzed}",
             "Tỷ lệ bên dưới chỉ tính trong ý kiến không trung lập, không phải toàn bộ khách hàng."]
    for aspect in ASPECTS:
        data = analysis.aspect_scores.get(aspect) or {}
        pos, neg = _number(data.get("pos"), True) or 0, _number(data.get("neg"), True) or 0
        total = pos + neg
        note = f"{round(100 * pos / total)}% tích cực trong {total} ý kiến không trung lập ({pos} khen / {neg} chê)" if total else "Chưa có ý kiến tích cực/tiêu cực để tính tỷ lệ"
        if 0 < total < 10:
            note += " · mẫu còn ít"
        lines.append(f"- **{ASPECTS_VI[aspect]}**: {note}")
    if analysis.sample_reviews:
        lines.append("\n### Một số đánh giá mẫu (không đại diện toàn bộ đánh giá)")
        for review in analysis.sample_reviews[:3]:
            content = " ".join(str(review.get("content") or "").split())[:600]
            lines.append(f"> Rating: {review.get('rating') or 'Chưa rõ'} — {content}")
    for warning in analysis.warnings:
        lines.append(f"\nLưu ý: {warning}")
    return "\n\n".join(lines)


def analysis_to_document(analysis: ProductAnalysis) -> dict:
    metadata = {"product_id": int(analysis.product_id), "name": analysis.name, "price": analysis.price,
                "rating_average": analysis.rating, "review_count": analysis.review_count,
                "product_url": analysis.url, "category_name": analysis.category}
    for aspect, score in analysis.aspect_scores.items():
        for key in ("pos", "neg", "score"):
            metadata[f"absa_{aspect}_{key}"] = score.get(key)
    return {"id": f"url_{analysis.product_id}", "doc_type": "product_analysis", "score": 0.0,
            "text": format_analysis_report(analysis), "metadata": metadata}


async def analyze_url(url: str, progress_callback=None, absa_model="logreg", user_query="", *, force_refresh=False) -> str:
    analysis = await analyze_product(url, progress_callback, absa_model, force_refresh=force_refresh)
    report = format_analysis_report(analysis)
    if user_query and user_query.strip() != url.strip():
        from runtime.chatbot import llm
        try:
            answer = await asyncio.to_thread(llm.ask_url_recommendation, user_query, report)
            report += "\n\n## Nhận xét\n\n" + answer
        except llm.LLMError as exc:
            report += f"\n\nChưa tạo được nhận xét LLM: {exc}"
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a Tiki URL without loading the chat embedding model")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", choices=ABSA_MODELS, default="logreg")
    parser.add_argument("--refresh", action="store_true", help="Bypass Redis/KB and fetch Tiki directly")
    parser.add_argument("--question", default="", help="Optional Google LLM question about the report")
    args = parser.parse_args()
    try:
        print(asyncio.run(analyze_url(args.url, absa_model=args.model, user_query=args.question,
                                      force_refresh=args.refresh)))
    except (AnalysisError, ValueError) as exc:
        parser.exit(1, f"Không phân tích được sản phẩm: {exc}\n")
