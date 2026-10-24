from __future__ import annotations
import os
from typing import Optional, List, Dict, Tuple, Any
"""
Module 3: Get Product Reviews from Tiki API V2
Input : products_list.csv
Output: products_reviews.csv  (only reviews WITH content)
Fields: product_id, customer_name, rating, content, created_at
"""

import requests
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("tiki_crawler.module3")

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

REVIEWS_URL = "https://tiki.vn/api/v2/reviews"
INPUT_FILE = "products_list.csv"
OUTPUT_FILE = "products_reviews.csv"
REVIEW_FIELDS = ["product_id", "customer_name", "rating", "content", "created_at"]

MAX_REVIEW_PAGES = 10   # safety cap per product
REVIEWS_PER_PAGE = 20


def _fmt_date(ts: int | Optional[str]) -> str:
    """Convert Unix timestamp or ISO string to readable date."""
    if not ts:
        return ""
    try:
        if isinstance(ts, int) or (isinstance(ts, str) and ts.isdigit()):
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)
    except Exception:
        return str(ts)


def fetch_reviews_page(product_id: str, page: int) -> tuple[list[dict], int]:
    """
    Fetch one page of reviews.
    Returns (list_of_reviews, total_pages).
    """
    params = {
        "product_id": product_id,
        "page": page,
        "limit": REVIEWS_PER_PAGE,
        "sort": "score|desc,id|desc,stars|all",
    }
    try:
        r = requests.get(REVIEWS_URL, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        reviews = data.get("data", [])
        paging = data.get("paging") or {}
        total = paging.get("last_page") or paging.get("total_pages") or 1
        return reviews, int(total)
    except requests.exceptions.Timeout:
        logger.warning(f"    Page {page}: timeout")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"    Page {page}: HTTP {e}")
    except Exception as e:
        logger.error(f"    Page {page}: error – {e}")
    return [], 1


def crawl_reviews_for_product(product_id: str) -> List[dict]:
    """Crawl all review pages for a single product, keeping only content-filled reviews."""
    collected: list[dict] = []
    page = 1

    while page <= MAX_REVIEW_PAGES:
        reviews, total_pages = fetch_reviews_page(product_id, page)

        if not reviews:
            break

        for rev in reviews:
            content = (rev.get("content") or "").strip()
            if not content:          # skip empty-content reviews
                continue
            collected.append({
                "product_id": product_id,
                "customer_name": (rev.get("created_by") or {}).get("name", ""),
                "rating": rev.get("rating", ""),
                "content": content,
                "created_at": _fmt_date(rev.get("created_at")),
            })

        logger.info(f"    Page {page}/{min(total_pages, MAX_REVIEW_PAGES)}: "
                    f"+{len([r for r in reviews if (r.get('content') or '').strip()])} reviews with content")

        if page >= total_pages:
            break
        page += 1
        time.sleep(0.6)

    return collected


def get_product_reviews(input_file: str = INPUT_FILE, output_dir: str = ".") -> str:
    """
    Read product IDs from CSV, crawl reviews, save those with content.
    Returns path to output CSV.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir) / OUTPUT_FILE

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as f:
        products = list(csv.DictReader(f))

    logger.info(f"[Module 3] Crawling reviews for {len(products)} products")

    all_reviews: list[dict] = []

    for i, p in enumerate(products, 1):
        pid = p.get("product_id", "").strip()
        if not pid:
            continue

        logger.info(f"  [{i}/{len(products)}] Reviews for product {pid}")
        reviews = crawl_reviews_for_product(pid)
        all_reviews.extend(reviews)
        logger.info(f"    → {len(reviews)} reviews with content collected")
        time.sleep(0.8)

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(all_reviews)

    logger.info(f"[Module 3] Done – {len(all_reviews)} reviews saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    get_product_reviews()
