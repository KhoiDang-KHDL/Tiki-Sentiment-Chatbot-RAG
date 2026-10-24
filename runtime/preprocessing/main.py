"""CLI for cleaning raw crawl CSVs without creating sentiment labels."""
import argparse
from pathlib import Path

import pandas as pd

from runtime.crawling.storage import BASE_DIR, RAW_DIR, check_output, write_rows
from .clean_detail import clean_detail
from .clean_review import clean_review


def preprocess_files(raw_dir=RAW_DIR, output_dir=BASE_DIR / 'data' / 'processed', *, overwrite=False):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    targets = [output_dir / 'processed_details.csv', output_dir / 'processed_products_reviews.csv']
    for target in targets:
        check_output(target, overwrite)
    # Load both before writing either file, so bad inputs do not leave half a run.
    details = clean_detail(pd.read_csv(raw_dir / 'products_detail.csv', dtype={'product_id': str}))
    reviews = clean_review(pd.read_csv(raw_dir / 'products_reviews.csv', dtype={'product_id': str}))
    for target, frame in zip(targets, (details, reviews)):
        write_rows(target, list(frame.columns), frame.to_dict('records'), overwrite)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--output', type=Path, default=BASE_DIR / 'data' / 'processed')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        for path in preprocess_files(args.raw_dir, args.output, overwrite=args.overwrite):
            print(path)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Preprocessing failed: {exc}\n')


if __name__ == '__main__':
    main()
