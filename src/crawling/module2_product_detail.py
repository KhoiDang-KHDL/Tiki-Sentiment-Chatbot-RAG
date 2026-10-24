from __future__ import annotations
import os
from typing import Optional, List, Dict, Tuple, Any
"""
Module 2: Get Product Detail from Tiki API V2
Input : products_list.csv
Output: products_detail.csv  (only products with reviews content)
"""

import requests
import csv
import time
import logging
from pathlib import Path

logger = logging.getLogger("tiki_crawler.module2")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": "https://tiki.vn/",
    "x-guest-token": os.getenv("TIKI_GUEST_TOKEN", ""),
}

DETAIL_URL = "https://tiki.vn/api/v2/products/{product_id}"
REVIEWS_SUMMARY_URL = "https://tiki.vn/api/v2/reviews"
INPUT_FILE = "products_list.csv"
OUTPUT_FILE = "products_detail.csv"

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


def fetch_product_detail(product_id: int | str) -> Optional[dict]:
    """Call product detail endpoint. Returns raw JSON dict or None."""
    url = DETAIL_URL.format(product_id=product_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        logger.warning(f"  {product_id}: timeout – skipping")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"  {product_id}: HTTP {e} – skipping")
    except Exception as e:
        logger.error(f"  {product_id}: error – {e}")
    return None


def fetch_ai_review_summary(product_id: int | str) -> str:
    """Try to fetch AI-generated review summary from reviews endpoint."""
    try:
        params = {"product_id": product_id, "page": 1, "limit": 1}
        r = requests.get(REVIEWS_SUMMARY_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Tiki sometimes returns ai_generated_summary or review_summary inside the reviews meta
        ai = (
            data.get("ai_generated_summary")
            or data.get("meta", {}).get("review_summary")
            or data.get("meta", {}).get("ai_summary")
            or ""
        )
        return _safe(ai)
    except Exception:
        return ""


def extract_row(pid: str, raw: dict) -> dict:
    """Flatten the raw product JSON into our target fields."""
    brand = raw.get("brand") or {}
    seller = raw.get("current_seller") or {}
    category = raw.get("categories") or {}
    stock = raw.get("stock_item") or {}

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
    """Quick check – return True only if this product has ≥1 review with content."""
    try:
        params = {"product_id": product_id, "page": 1, "limit": 5}
        r = requests.get(REVIEWS_SUMMARY_URL, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for rev in data.get("data", []):
            if rev.get("content", "").strip():
                return True
    except Exception:
        pass
    return False


def get_product_detail(input_file: str = INPUT_FILE, output_dir: str = ".") -> str:
    """
    Read product IDs from CSV, fetch details, keep only those with review content.
    Saves to products_detail.csv and returns its path.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir) / OUTPUT_FILE

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load product list
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        products = list(reader)

    logger.info(f"[Module 2] Processing {len(products)} products from {input_path}")

    rows: list[dict] = []

    for i, p in enumerate(products, 1):
        pid = p.get("product_id", "").strip()
        if not pid:
            continue

        logger.info(f"  [{i}/{len(products)}] Product {pid}")

        # Check if product has review content (to skip early)
        if not has_reviews(pid):
            logger.info(f"    → no review content, skipping")
            time.sleep(0.4)
            continue

        raw = fetch_product_detail(pid)
        if not raw:
            time.sleep(0.5)
            continue

        row = extract_row(pid, raw)

        # Try AI summary
        row["ai_review_summary"] = fetch_ai_review_summary(pid)

        rows.append(row)
        time.sleep(0.8)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[Module 2] Done – {len(rows)} products saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    get_product_detail()
