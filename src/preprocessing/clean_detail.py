from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
from pathlib import Path
import re
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = BASE_DIR / "data/raw/products_detail.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/processed_details.csv"

def clean_html(text):
    if pd.isna(text) or not str(text).strip():
        return ""

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_text(text):
    if pd.isna(text) or not str(text).strip():
        return ""

    spam_patterns = [
        "liên hệ shop",
        "hình ảnh mang tính minh họa",
        "vui lòng liên hệ",
        "shop cam kết"
    ]
    for p in spam_patterns:
        text = text.replace(p, "")

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def run_clean_detail():
    df = pd.read_csv(RAW_PATH)

    keep_cols = ['product_id',
                 'sku',
                 'name',
                 'brand_id',
                 'brand_name',
                 'category_id',
                 'category_name',
                 'price',
                 'list_price',
                 'discount',
                 'discount_rate',
                 'short_description',
                 'specifications',
                 'seller_id',
                 'seller_sku',
                 'seller_name',
                 'inventory_status',
                 'thumbnail_url',
                 'product_url',
                 'url_path',
                 'rating_average',
                 'review_count'
                 ]
    df = df[keep_cols].copy()

    df['short_description'] = df['short_description'].apply(clean_html).apply(clean_text)
    df['specifications'] = df['specifications'].apply(clean_html).apply(clean_text)

    df = df.rename(columns={
        "rating_average": "rating_from_detail",
        "review_count": "review_count_from_detail"})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    run_clean_detail()
