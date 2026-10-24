from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
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

    df = df.drop_duplicates(subset=['clean_content'])

    return df
