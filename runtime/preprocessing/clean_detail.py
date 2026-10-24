"""Clean exported product descriptions without changing raw crawl files."""
from html import unescape
from html.parser import HTMLParser
import re

import pandas as pd


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_html(text):
    if not isinstance(text, str):
        return ''
    parser = _TextParser()
    parser.feed(text)
    return re.sub(r'\s+', ' ', unescape(' '.join(parser.parts))).strip()


def clean_text(text):
    if not isinstance(text, str):
        return ''
    for phrase in ('liên hệ shop', 'hình ảnh mang tính minh họa', 'vui lòng liên hệ', 'shop cam kết'):
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def clean_detail(df):
    """Keep all metadata needed by the current KB builder (no rating renames)."""
    if 'product_id' not in df.columns:
        raise ValueError('Product details must contain product_id')
    result = df.copy()
    for column in ('short_description', 'specifications', 'ai_review_summary'):
        if column in result:
            result[column] = result[column].apply(clean_html).apply(clean_text)
    return result.drop_duplicates(subset=['product_id'])
