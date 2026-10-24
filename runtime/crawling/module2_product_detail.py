from __future__ import annotations
"""
Module 2: Get Product Detail from Tiki API V2
Input : products_list.csv
Output: products_detail.csv  (only products with reviews content)
"""

import time
import logging
from pathlib import Path

logger = logging.getLogger("tiki_crawler.module2")

from .client import CrawlError, get_json, validate_product_id
from .storage import RAW_DIR, check_output, read_product_ids, write_rows

# All fields we try to extract from the product detail API
DETAIL_FIELDS = [
    "product_id", "name", "short_name", "sku", "price", "list_price",
    "discount", "discount_rate", "rating_average", "review_count",
    "order_count", "favourite_count", "thumbnail_url", "brand_name",
    "brand_id", "seller_name", "seller_id", "seller_sku",
    "category_name", "category_id", "short_description",
    "specifications", "inventory_status", "stock_item_qty",
    "stock_item_max_sale_qty", "has_ebook", "is_fresh", "is_genuine",
    "url_path", "product_url",
    # AI review summary (if available)
    "ai_review_summary",
]


def _safe(val) -> str:
    """Stringify any value safely."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        import json
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def fetch_product_detail(product_id: int | str) -> dict:
    """Fetch raw JSON. Raise CrawlError rather than silently returning partial data."""
    data = get_json(f"products/{validate_product_id(product_id)}")
    if not data.get("name"):
        raise CrawlError("Không tìm thấy thông tin sản phẩm trên Tiki.")
    return data


def fetch_ai_review_summary(product_id) -> str:
    data = get_json("reviews", {"product_id": validate_product_id(product_id), "page": 1, "limit": 1})
    meta = data.get("meta") or {}
    return _safe(data.get("ai_generated_summary") or meta.get("review_summary") or meta.get("ai_summary"))


def extract_row(pid: str, raw: dict) -> dict:
    """Flatten the raw product JSON into our target fields."""
    brand = raw.get("brand") or {}
    brand = brand if isinstance(brand, dict) else {}
    seller = raw.get("current_seller") or {}
    seller = seller if isinstance(seller, dict) else {}
    category = raw.get("categories") or {}
    if isinstance(category, list):
        category = category[-1] if category else {}
    category = category if isinstance(category, dict) else {}
    stock = raw.get("stock_item") or {}
    stock = stock if isinstance(stock, dict) else {}

    # Specifications as compact text
    specs_raw = raw.get("specifications") or []
    specs_text = "; ".join(
        f"{attr.get('name','')}: {attr.get('value','')}"
        for group in specs_raw
        for attr in (group.get("attributes") or [])
        if attr.get("value")
    )

    return {
        "product_id": pid,
        "name": _safe(raw.get("name")),
        "short_name": _safe(raw.get("short_name")),
        "sku": _safe(raw.get("sku")),
        "price": _safe(raw.get("price")),
        "list_price": _safe(raw.get("list_price")),
        "discount": _safe(raw.get("discount")),
        "discount_rate": _safe(raw.get("discount_rate")),
        "rating_average": _safe(raw.get("rating_average")),
        "review_count": _safe(raw.get("review_count")),
        "order_count": _safe(raw.get("order_count")),
        "favourite_count": _safe(raw.get("favourite_count")),
        "thumbnail_url": _safe(raw.get("thumbnail_url")),
        "brand_name": _safe(brand.get("name")),
        "brand_id": _safe(brand.get("id")),
        "seller_name": _safe(seller.get("name")),
        "seller_id": _safe(seller.get("id")),
        "seller_sku": _safe(seller.get("sku")),
        "category_name": _safe(category.get("name")),
        "category_id": _safe(category.get("id")),
        "short_description": _safe(raw.get("short_description")),
        "specifications": specs_text,
        "inventory_status": _safe(raw.get("inventory_status")),
        "stock_item_qty": _safe(stock.get("qty")),
        "stock_item_max_sale_qty": _safe(stock.get("max_sale_qty")),
        "has_ebook": _safe(raw.get("has_ebook")),
        "is_fresh": _safe(raw.get("is_fresh")),
        "is_genuine": _safe(raw.get("is_genuine")),
        "url_path": _safe(raw.get("url_path")),
        "product_url": f"https://tiki.vn/{raw.get('url_path','')}" if raw.get("url_path") else "",
        "ai_review_summary": "",  # filled later
    }


def has_reviews(product_id: str) -> bool:
    from .module3_product_reviews import fetch_reviews_page
    reviews, _ = fetch_reviews_page(product_id, 1)
    return any(isinstance(review, dict) and str(review.get("content") or "").strip() for review in reviews)


def get_product_detail(input_file=RAW_DIR / "products_list.csv", output_dir=RAW_DIR, *,
                       only_with_reviews=True, include_ai_summary=False, delay=0.8, overwrite=False):
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    output = check_output(Path(output_dir) / "products_detail.csv", overwrite)
    rows = []
    for pid in read_product_ids(input_file):
        if not only_with_reviews or has_reviews(pid):
            row = extract_row(pid, fetch_product_detail(pid))
            if include_ai_summary:
                row["ai_review_summary"] = fetch_ai_review_summary(pid)
            rows.append(row)
        time.sleep(delay)
    return write_rows(output, DETAIL_FIELDS, rows, overwrite)
