"""Discover product IDs by category and export products_list.csv."""
import time

from .client import CrawlError, get_json, validate_product_id
from .storage import RAW_DIR, check_output, write_rows


def fetch_product_page(category_id, page, limit=40):
    category_id = validate_product_id(category_id)
    if page < 1 or not 1 <= limit <= 40:
        raise ValueError("page must be positive; limit must be between 1 and 40")
    data = get_json("products", {"category": category_id, "page": page, "limit": limit,
                                 "sort": "top_seller", "urlKey": ""})
    products = data.get("data")
    if not isinstance(products, list):
        raise CrawlError("Dữ liệu danh sách sản phẩm không đúng định dạng.")
    return products


def get_product_list(category_id, limit_page=3, output_dir=RAW_DIR, *, delay=0.8, overwrite=False):
    from pathlib import Path
    if limit_page < 1 or delay < 0:
        raise ValueError("limit_page must be positive and delay nonnegative")
    output = check_output(Path(output_dir) / "products_list.csv", overwrite)
    rows, seen = [], set()
    for page in range(1, limit_page + 1):
        products = fetch_product_page(category_id, page)
        if not products:
            break
        for product in products:
            if not isinstance(product, dict) or not product.get("id"):
                continue
            pid = validate_product_id(product["id"])
            if pid not in seen:
                seen.add(pid)
                rows.append({"product_id": pid, "product_url": f"https://tiki.vn/{product.get('url_path') or f'product-p{pid}.html'}"})
        if page < limit_page:
            time.sleep(delay)
    return write_rows(output, ["product_id", "product_url"], rows, overwrite)
