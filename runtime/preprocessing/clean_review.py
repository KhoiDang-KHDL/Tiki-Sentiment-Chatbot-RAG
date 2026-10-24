"""Reference review normalization, shared by batch processing and inference.

Keep this transformation aligned with model training; preserve raw content for
display. Existing labeled data may contain additional corrections not implemented
by the reference cleaner.
"""
from __future__ import annotations
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text).lower()

    abbreviations = {
        r'\bsp\b': 'sản phẩm',
        r'\bksd\b': 'không sử dụng',
        r'\btks\b|\bthanks\b': 'cảm ơn',
        r'\bđc\b': 'được',
        r'\bclg\b': 'chất lượng',
        r'\bhđ\b': 'hướng dẫn',
        r'\bnv\b': 'nhân viên',
        r'\bko\b|\bk\b': 'không',
        r'\bj\b': 'gì',
        r'\bship\b': 'giao hàng',
        r'\bre\b': 'rẻ',
        r'\btl\b': 'trả lời',
        r'\bvs\b': 'với'
    }

    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)

    text = text.replace("\n", " ")
    text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)
    text = re.sub(r'(?<!\d)/|/(?!\d)', ' ', text)

    text = re.sub(r'[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]', '', text)

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    return text

def clean_review(df, column_name='content'):
    df = df.copy()

    df['clean_content'] = df[column_name].apply(clean_text)

    df = df[df['clean_content'] != ""]
    df = df.dropna(subset=['clean_content'])

    # The same sentence on two products is still two distinct observations.
    keys = ['product_id', 'clean_content'] if 'product_id' in df.columns else ['clean_content']
    df = df.drop_duplicates(subset=keys)

    return df


def prepare_reviews(reviews):
    """Preserve raw fields for display; filter/deduplicate normalized model text."""
    rows, seen = [], set()
    for review in reviews:
        if not isinstance(review, dict):
            continue
        cleaned = clean_text(review.get('content'))
        key = (str(review.get('product_id', '')), cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        rows.append({**review, 'clean_content': cleaned})
    return rows
