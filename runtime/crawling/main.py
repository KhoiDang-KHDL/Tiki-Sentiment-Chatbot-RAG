"""Batch Tiki crawler: category -> list -> details + reviews -> raw CSVs."""
import argparse
from pathlib import Path

from .client import CrawlError
from .module1_product_list import get_product_list
from .module2_product_detail import get_product_detail
from .module3_product_reviews import get_product_reviews
from .storage import RAW_DIR, check_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', type=int, default=1846)
    parser.add_argument('--pages', type=int, default=3, help='Number of product-list pages')
    parser.add_argument('--review-pages', type=int, choices=range(1, 11), default=10)
    parser.add_argument('--module', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--output', type=Path, default=RAW_DIR)
    parser.add_argument('--input-csv', type=Path)
    parser.add_argument('--delay', type=float, default=0.8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include-without-reviews', action='store_true')
    parser.add_argument('--ai-summary', action='store_true')
    args = parser.parse_args()
    if args.category < 1 or args.pages < 1 or args.delay < 0:
        parser.error('category/pages must be positive; delay must be nonnegative')
    selected = (1, 2, 3) if args.module == 0 else (args.module,)
    names = {1: 'products_list.csv', 2: 'products_detail.csv', 3: 'products_reviews.csv'}
    try:
        # Check all outputs before making network calls or modifying any file.
        for module in selected:
            check_output(args.output / names[module], args.overwrite)
        products_csv = args.input_csv or args.output / 'products_list.csv'
        if 1 in selected:
            products_csv = get_product_list(args.category, args.pages, args.output,
                                            delay=args.delay, overwrite=args.overwrite)
            print(products_csv)
        if 2 in selected:
            print(get_product_detail(products_csv, args.output, delay=args.delay, overwrite=args.overwrite,
                only_with_reviews=not args.include_without_reviews, include_ai_summary=args.ai_summary))
        if 3 in selected:
            print(get_product_reviews(products_csv, args.output, max_pages=args.review_pages,
                                      delay=args.delay, overwrite=args.overwrite))
    except (CrawlError, OSError, ValueError) as exc:
        parser.exit(1, f'Crawl failed: {exc}\n')


if __name__ == '__main__':
    main()
