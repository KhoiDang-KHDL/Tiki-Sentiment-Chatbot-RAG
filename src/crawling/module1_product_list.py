from __future__ import annotations
import os
from typing import Optional, List, Dict, Tuple, Any
"""
Module 1: Get Product List from Tiki API V2
Input : Category ID + limit_page
Output: products_list.csv (product_id, product_url)
"""

import requests
import csv
import time
import logging
from pathlib import Path

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("tiki_crawler.module1")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://tiki.vn/",
    "x-guest-token": os.getenv("TIKI_GUEST_TOKEN", ""),
}

BASE_URL = "https://tiki.vn/api/v2/products"
OUTPUT_FILE = "products_list.csv"


def fetch_product_page(category_id: int, page: int, limit: int = 40) -> List[dict]:
    """Fetch one page of products for a category. Returns list of product dicts."""
    params = {
        "category": category_id,
        "page": page,
        "limit": limit,
        "sort": "top_seller",
        "urlKey": "",
    }
    try:
        response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status() # api tự ném lỗi
        data = response.json()
        products = data.get("data", [])
        logger.info(f"  Page {page}: fetched {len(products)} products")
        return products
    except requests.exceptions.Timeout:
        logger.warning(f"  Page {page}: request timed out – skipping")
        return []
    except requests.exceptions.HTTPError as e:
        logger.warning(f"  Page {page}: HTTP error {e} – skipping")
        return []
    except Exception as e:
        logger.error(f"  Page {page}: unexpected error – {e}")
        return []


def get_product_list(category_id: int, limit_page: int, output_dir: str = ".") -> str:
    """
    Crawl `limit_page` pages of products for `category_id`.
    Saves product_id + product_url to CSV and returns the output path.
    """
    output_path = Path(output_dir) / OUTPUT_FILE
    logger.info(f"[Module 1] Category={category_id}, pages=1–{limit_page}")

    rows: list[dict] = []
    seen_ids: set = set()

    for page in range(1, limit_page + 1):
        products = fetch_product_page(category_id, page)
        for p in products:
            pid = p.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            url_path = p.get("url_path", "")
            product_url = f"https://tiki.vn/{url_path}" if url_path else ""
            rows.append({"product_id": pid, "product_url": product_url})
        time.sleep(0.8)          # polite delay

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "product_url"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[Module 1] Done – {len(rows)} products saved to {output_path}")
    return str(output_path)


# ── Stand-alone usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Example: category 1846 = Điện thoại
    get_product_list(category_id=1846, limit_page=3)
