from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Tiki Crawler – Main Runner
Orchestrates Module 1 → Module 2 → Module 3
Usage:
    python -m src.crawling.main --category 1883 --pages 3 (in root)
    python -m src.crawling.main --category 1883 --pages 3 --module 1   # run only module 1
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parents[2] / "logs" # đi lên 2 cấp từ main.py: từ main.py trong crawler, từ crawler -> src, src -> mlops...
LOG_DIR.mkdir(exist_ok=True)
log_filename = LOG_DIR / f"tiki_crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("tiki_crawler.main")

# ── Import modules ─────────────────────────────────────────────────────────────
from .module1_product_list import get_product_list
from .module2_product_detail import get_product_detail
from .module3_product_reviews import get_product_reviews


def main():
    parser = argparse.ArgumentParser(
        description="Tiki Crawler – crawl product list, details, and reviews via API V2"
    )
    parser.add_argument(
        "--category", type=int, default=1846,
        help="Tiki category ID (default: 1846 – Điện thoại)"
    )
    parser.add_argument(
        "--pages", type=int, default=3,
        help="Number of pages to fetch from product list (default: 3)"
    )
    parser.add_argument(
        "--module", type=int, choices=[1, 2, 3], default=0,
        help="Run only a specific module (1/2/3). Default=0 runs all."
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--input-csv", type=str, default=None,
        help="Path to products_list.csv (used by module 2/3 when running standalone)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  TIKI CRAWLER STARTED")
    logger.info(f"  Category ID : {args.category}")
    logger.info(f"  Pages       : {args.pages}")
    logger.info(f"  Output dir  : {output_dir.resolve()}")
    logger.info(f"  Log file    : {log_filename.resolve()}")
    logger.info("=" * 60)

    run_all = args.module == 0
    products_csv = args.input_csv or str(output_dir / "products_list.csv")

    try:
        # ── Module 1 ──────────────────────────────────────────────────────────
        if run_all or args.module == 1:
            logger.info("\n▶ MODULE 1: Get Product List")
            t0 = time.time()
            products_csv = get_product_list(
                category_id=args.category,
                limit_page=args.pages,
                output_dir=str(output_dir),
            )
            logger.info(f"  Completed in {time.time()-t0:.1f}s → {products_csv}\n")

        # ── Module 2 ──────────────────────────────────────────────────────────
        if run_all or args.module == 2:
            logger.info("\n▶ MODULE 2: Get Product Detail")
            t0 = time.time()
            detail_csv = get_product_detail(
                input_file=products_csv,
                output_dir=str(output_dir),
            )
            logger.info(f"  Completed in {time.time()-t0:.1f}s → {detail_csv}\n")

        # ── Module 3 ──────────────────────────────────────────────────────────
        if run_all or args.module == 3:
            logger.info("\n▶ MODULE 3: Get Product Reviews")
            t0 = time.time()
            reviews_csv = get_product_reviews(
                input_file=products_csv,
                output_dir=str(output_dir),
            )
            logger.info(f"  Completed in {time.time()-t0:.1f}s → {reviews_csv}\n")

    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  ALL DONE ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
