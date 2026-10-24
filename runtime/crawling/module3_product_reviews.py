"""Fetch review pages, preserve raw text and export batch reviews."""
from datetime import datetime, timezone
from pathlib import Path
import time

from .client import CrawlError, get_json, validate_product_id
from .storage import RAW_DIR, check_output, read_product_ids, write_rows

REVIEWS_PER_PAGE = 20
MAX_REVIEW_PAGES = 10
REVIEW_FIELDS = ["product_id", "customer_name", "rating", "content", "created_at"]


def fetch_reviews_page(product_id, page):
    pid = validate_product_id(product_id)
    if page < 1:
        raise ValueError("page must be positive")
    data = get_json("reviews", {"product_id": pid, "page": page, "limit": REVIEWS_PER_PAGE,
                                "sort": "score|desc,id|desc,stars|all"})
    reviews = data.get("data")
    if not isinstance(reviews, list):
        raise CrawlError("Dữ liệu đánh giá Tiki không đúng định dạng.")
    paging = data.get("paging") or {}
    total = paging.get("last_page") or paging.get("total_pages")
    try:
        total = int(total)
        total = total if total > 0 else None
    except (TypeError, ValueError, OverflowError):
        total = None
    return reviews, total


def _fmt_date(value):
    if value is None or value == "":
        return ""
    try:
        return datetime.fromtimestamp(int(value), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OverflowError, OSError):
        return str(value)


def crawl_reviews_for_product(product_id, max_pages=MAX_REVIEW_PAGES, *, delay=0.6):
    pid = validate_product_id(product_id)
    if not 1 <= max_pages <= MAX_REVIEW_PAGES or delay < 0:
        raise ValueError("max_pages must be between 1 and 10; delay must be nonnegative")
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        reviews, last_page = fetch_reviews_page(pid, page)
        if not reviews:
            break
        for review in reviews[:REVIEWS_PER_PAGE]:
            if not isinstance(review, dict):
                continue
            content = str(review.get("content") or "").strip()
            key = str(review.get("id") or content)
            if not content or key in seen:
                continue
            seen.add(key)
            author = review.get("created_by") or {}
            rows.append({"product_id": pid, "customer_name": author.get("name", "") if isinstance(author, dict) else "",
                         "rating": review.get("rating"), "content": content,
                         "created_at": _fmt_date(review.get("created_at"))})
        if last_page is not None and page >= last_page:
            break
        if page < max_pages:
            time.sleep(delay)
    return rows


def get_product_reviews(input_file=RAW_DIR / "products_list.csv", output_dir=RAW_DIR, *,
                        max_pages=MAX_REVIEW_PAGES, delay=0.6, overwrite=False):
    output = check_output(Path(output_dir) / "products_reviews.csv", overwrite)
    rows = []
    for pid in read_product_ids(input_file):
        rows.extend(crawl_reviews_for_product(pid, max_pages=max_pages, delay=delay))
        time.sleep(delay)
    return write_rows(output, REVIEW_FIELDS, rows, overwrite)
