from __future__ import annotations
import os
from typing import Optional, List, Dict, Tuple, Any
"""
Scheduler: Crawl sản phẩm có sự thay đổi về số bình luận trong 15 ngày gần nhất.

Flow:
    1. Đọc products_list.csv → lấy danh sách product_id cần theo dõi
    2. Với mỗi sản phẩm: gọi API lấy review_count hiện tại
    3. So sánh với review_count lần crawl trước (lưu trong state.json)
    4. Nếu có thay đổi → crawl review mới, bỏ qua review trống
    5. Lưu kết quả ra file CSV theo ngày: reviews_YYYYMMDD.csv
    6. Cập nhật state.json
    7. Lập lịch chạy lại theo APScheduler

Lưu vết (state.json):
    {
        "product_id": {
            "review_count": 120,
            "last_crawled": "2026-04-14 02:00:00"
        },
        ...
    }

Cách chạy:
    python -m src.crawling.scheduler              # chạy ngay + lập lịch
    python -m src.crawling.scheduler --run-now    # chạy 1 lần rồi thoát (debug)
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # sentiment-chatbot-mlops/
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
STATE_FILE = RAW_DIR / "crawl_state.json"           # lưu vết review_count

LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_filename = LOG_DIR / f"scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("tiki_crawler.scheduler")

# ── Constants ─────────────────────────────────────────────────────────────────
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
REVIEWS_URL = "https://tiki.vn/api/v2/reviews"
REVIEWS_PER_PAGE = 20
MAX_REVIEW_PAGES = 10
REVIEW_FIELDS = ["product_id", "customer_name", "rating", "content", "created_at"]

# Chỉ lấy review trong vòng 15 ngày gần nhất
LOOKBACK_DAYS = 15


# ── State helpers ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Đọc file state.json. Trả về dict rỗng nếu chưa có."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Không đọc được state.json: {e} → reset state")
    return {}


def save_state(state: dict) -> None:
    """Ghi state dict ra state.json."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu state → {STATE_FILE}")


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_review_count(product_id: str) -> Optional[int]:
    """Lấy review_count hiện tại của sản phẩm từ product detail API."""
    url = DETAIL_URL.format(product_id=product_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return int(data.get("review_count") or 0)
    except requests.exceptions.Timeout:
        logger.warning(f"  {product_id}: timeout khi lấy review_count")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"  {product_id}: HTTP {e} khi lấy review_count")
    except Exception as e:
        logger.error(f"  {product_id}: lỗi không xác định – {e}")
    return None


def _fmt_date(ts: int | Optional[str]) -> str:
    """Chuyển Unix timestamp hoặc ISO string thành datetime string."""
    if not ts:
        return ""
    try:
        if isinstance(ts, int) or (isinstance(ts, str) and str(ts).isdigit()):
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)
    except Exception:
        return str(ts)


def _parse_dt(ts: int | Optional[str]) -> datetime | None:
    """Parse timestamp thành datetime object để so sánh."""
    if not ts:
        return None
    try:
        if isinstance(ts, int) or (isinstance(ts, str) and str(ts).isdigit()):
            return datetime.utcfromtimestamp(int(ts))
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def fetch_new_reviews(product_id: str, cutoff: datetime) -> List[dict]:
    """
    Crawl review của 1 sản phẩm.
    Chỉ lấy review:
        - Có content (không rỗng)
        - created_at >= cutoff (trong vòng LOOKBACK_DAYS ngày)
    Dừng sớm khi gặp review cũ hơn cutoff.
    """
    collected = []
    page = 1

    while page <= MAX_REVIEW_PAGES:
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
        except requests.exceptions.Timeout:
            logger.warning(f"    Page {page}: timeout")
            break
        except requests.exceptions.HTTPError as e:
            logger.warning(f"    Page {page}: HTTP {e}")
            break
        except Exception as e:
            logger.error(f"    Page {page}: lỗi – {e}")
            break

        reviews = data.get("data", [])
        if not reviews:
            break

        stop_early = False
        for rev in reviews:
            created_dt = _parse_dt(rev.get("created_at"))

            # Dừng sớm nếu review cũ hơn cutoff
            if created_dt and created_dt < cutoff:
                stop_early = True
                break

            content = (rev.get("content") or "").strip()
            if not content:     # bỏ qua review trống
                continue

            collected.append({
                "product_id": product_id,
                "customer_name": (rev.get("created_by") or {}).get("name", ""),
                "rating": rev.get("rating", ""),
                "content": content,
                "created_at": _fmt_date(rev.get("created_at")),
            })

        paging = data.get("paging") or {}
        total_pages = int(paging.get("last_page") or paging.get("total_pages") or 1)
        logger.info(
            f"    Page {page}/{min(total_pages, MAX_REVIEW_PAGES)}: "
            f"+{len(collected)} reviews có content tính đến hiện tại"
        )

        if stop_early or page >= total_pages:
            break

        page += 1
        time.sleep(0.6)

    return collected


# ── Core job ──────────────────────────────────────────────────────────────────

def crawl_changed_products(products_csv: Optional[str] = None) -> None:
    """
    Job chính được lập lịch chạy.
    1. Đọc danh sách product_id từ products_list.csv
    2. Kiểm tra review_count có thay đổi không
    3. Crawl review mới (trong 15 ngày, bỏ review trống)
    4. Lưu ra reviews_YYYYMMDD.csv
    5. Cập nhật state.json
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().strftime("%Y%m%d")
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)

    logger.info("=" * 60)
    logger.info(f"  SCHEDULER JOB BẮT ĐẦU – {now_str}")
    logger.info(f"  Chỉ lấy review từ {cutoff.strftime('%Y-%m-%d')} trở đi")
    logger.info("=" * 60)

    # Xác định file products_list.csv
    csv_path = Path(products_csv) if products_csv else RAW_DIR / "products_list.csv"
    if not csv_path.exists():
        logger.error(f"Không tìm thấy {csv_path} – job bị bỏ qua")
        return

    # Đọc danh sách sản phẩm
    with open(csv_path, newline="", encoding="utf-8") as f:
        products = [row for row in csv.DictReader(f) if row.get("product_id", "").strip()]
    logger.info(f"Tổng sản phẩm cần kiểm tra: {len(products)}")

    # Load state cũ
    state = load_state()

    all_new_reviews: list[dict] = []
    changed_count = 0

    for i, p in enumerate(products, 1):
        pid = p["product_id"].strip()
        logger.info(f"[{i}/{len(products)}] Kiểm tra product {pid}")

        # Lấy review_count hiện tại
        current_count = fetch_review_count(pid)
        if current_count is None:
            time.sleep(0.5)
            continue

        # So sánh với lần trước
        prev_info = state.get(pid, {})
        prev_count = prev_info.get("review_count", -1)

        if current_count == prev_count:
            logger.info(f"  → Không có review mới (vẫn {current_count}) – bỏ qua")
            time.sleep(0.3)
            continue

        # Có thay đổi → crawl review mới
        logger.info(
            f"  → Thay đổi: {prev_count} → {current_count} "
            f"(+{current_count - prev_count if prev_count >= 0 else '?'})"
        )
        changed_count += 1

        new_reviews = fetch_new_reviews(pid, cutoff)
        logger.info(f"  → Thu được {len(new_reviews)} review có nội dung trong {LOOKBACK_DAYS} ngày")
        all_new_reviews.extend(new_reviews)

        # Cập nhật state cho sản phẩm này
        state[pid] = {
            "review_count": current_count,
            "last_crawled": now_str,
        }

        time.sleep(0.8)

    # Lưu kết quả ra file CSV theo ngày
    if all_new_reviews:
        output_path = RAW_DIR / f"reviews_{today_str}.csv"
        # Nếu file đã tồn tại trong ngày → append thêm vào
        file_exists = output_path.exists()
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_new_reviews)
        logger.info(f"Đã lưu {len(all_new_reviews)} reviews → {output_path}")
    else:
        logger.info("Không có review mới nào để lưu")

    # Lưu state
    save_state(state)

    logger.info("=" * 60)
    logger.info(
        f"  JOB HOÀN TẤT – {changed_count}/{len(products)} sản phẩm có thay đổi, "
        f"{len(all_new_reviews)} reviews mới"
    )
    logger.info("=" * 60)


# ── Scheduler setup ───────────────────────────────────────────────────────────

def run_scheduler() -> None:
    """Khởi động APScheduler, chạy job mỗi ngày lúc 2:00 AM."""
    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")

    scheduler.add_job(
        func=crawl_changed_products,
        trigger=IntervalTrigger(days=15, start_date=datetime.now()),
        id="tiki_crawl_changed",
        name="Crawl sản phẩm có review mới",
        misfire_grace_time=300,                  # cho phép trễ tối đa 5 phút
        replace_existing=True,
    )

    next_run = scheduler.get_jobs()[0].next_run_time
    logger.info(f"Scheduler đã khởi động. Lần chạy tiếp theo: {next_run}")
    logger.info("Nhấn Ctrl+C để dừng.\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler dừng theo yêu cầu.")
        scheduler.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiki Scheduler – crawl review có thay đổi")
    parser.add_argument(
        "--run-now", action="store_true",
        help="Chạy job ngay lập tức 1 lần rồi thoát (dùng để test)"
    )
    parser.add_argument(
        "--products-csv", type=str, default=None,
        help="Đường dẫn tới products_list.csv (mặc định: data/raw/products_list.csv)"
    )
    args = parser.parse_args()

    if args.run_now:
        logger.info("Chế độ --run-now: chạy 1 lần rồi thoát")
        crawl_changed_products(products_csv=args.products_csv)
    # else:
    #     # Chạy ngay 1 lần khi khởi động, sau đó lập lịch hàng ngày
    #     logger.info("Chạy job lần đầu ngay khi khởi động...")
    #     crawl_changed_products(products_csv=args.products_csv)
        # run_scheduler()
