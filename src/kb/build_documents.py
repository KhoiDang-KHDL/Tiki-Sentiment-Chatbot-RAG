from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Stage 2: Build documents cho Qdrant
3 loại per product: product_card, aspect_summary, review

Input : data/raw/products_detail.csv
        data/processed/labeled_processed_products_reviews.csv
        data/processed/product_aspect_scores.parquet
Output: data/processed/documents.jsonl
"""
import ast
import json
import re
import pandas as pd
from html import unescape
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
DETAIL_CSV   = BASE_DIR / "data/raw/products_detail.csv"
LABELED_CSV  = BASE_DIR / "data/processed/labeled_processed_products_reviews.csv"
SCORES_PAR   = BASE_DIR / "data/processed/product_aspect_scores.parquet"
OUTPUT_FILE  = BASE_DIR / "data/processed/documents.jsonl"

# ── Constants ──────────────────────────────────────────
ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]
ASPECT_VI = {
    "description": "mô tả",
    "quality":     "chất lượng",
    "packaging":   "đóng gói",
    "delivery":    "giao hàng",
    "service":     "dịch vụ",
    "price":       "giá cả",
}


# ── Helpers ────────────────────────────────────────────
def get_top_k(n_reviews: int) -> int:
    if n_reviews <= 3:
        return n_reviews
    if n_reviews <= 10:
        return 4
    if n_reviews <= 20:
        return 5
    if n_reviews <= 50:
        return 6
    if n_reviews <= 100:
        return 8
    return min(max(round(n_reviews * 0.08), 8), 20)

def clean_html(text) -> str:
    if not isinstance(text, str):
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fmt_price(n) -> str:
    try:
        return f"{int(n):,}đ".replace(",", ".")
    except:
        return "n/a"


def safe_int(v):
    try:
        return None if v != v else int(float(v))
    except:
        return None


def safe_float(v):
    try:
        return None if v != v else round(float(v), 2)
    except:
        return None


# ── Document builders ──────────────────────────────────
def build_product_card(det: pd.Series, scores: Optional[dict]) -> dict:
    pid = int(det["product_id"])
    lines = [
        f"Sản phẩm: {det.get('name', '')}",
        f"Thương hiệu: {det.get('brand_name', 'n/a')}",
        f"Danh mục: {det.get('category_name', 'n/a')}",
        f"Giá: {fmt_price(det.get('price'))}",
        f"Đánh giá: {det.get('rating_average', 'n/a')}/5 "
        f"({safe_int(det.get('review_count')) or 0} đánh giá)",
    ]
    desc = clean_html(det.get("short_description"))
    if desc:
        lines.append(f"Mô tả: {desc[:500]}")
    ai_sum = clean_html(det.get("ai_review_summary"))
    if ai_sum:
        lines.append(f"Tóm tắt đánh giá: {ai_sum[:500]}")

    return {
        "id":       f"prod_{pid}_card",
        "doc_type": "product_card",
        "content":  "\n".join(lines),
        "metadata": build_metadata(det, scores),
    }


def build_aspect_summary(det: pd.Series, scores: Optional[dict]) -> Optional[dict]:
    if not scores:
        return None
    pid = int(det["product_id"])
    lines = [f"Tổng hợp đánh giá '{det.get('name', '')}':"]
    has_data = False

    for asp in ASPECTS:
        pos   = int(scores.get(f"{asp}_pos") or 0)
        neg   = int(scores.get(f"{asp}_neg") or 0)
        neu   = int(scores.get(f"{asp}_neu") or 0)
        total = pos + neg + neu
        if total == 0:
            continue
        has_data = True
        pct_pos = round(100 * pos / total)
        pct_neg = round(100 * neg / total)
        verdict = (
            "đa số khen"       if pct_pos >= 70 else
            "khá tích cực"     if pct_pos >= 50 else
            "đa số chê"        if pct_neg >= 50 else
            "ý kiến trái chiều"
        )
        lines.append(
            f"- {ASPECT_VI[asp].capitalize()}: {verdict} "
            f"({pct_pos}% tích cực, {pct_neg}% tiêu cực / {total} ý kiến)"
        )
    if not has_data:
        return None

    return {
        "id":       f"prod_{pid}_aspects",
        "doc_type": "aspect_summary",
        "content":  "\n".join(lines),
        "metadata": build_metadata(det, scores),
    }


def build_reviews(det: pd.Series, reviews: pd.DataFrame, scores: Optional[dict]) -> List[dict]:
    pid = int(det["product_id"])
    if reviews.empty:
        return []

    reviews = reviews.copy()
    reviews["_len"] = reviews["content"].astype(str).str.len()

    def overall(label_str):
        try:
            labels = ast.literal_eval(label_str)
            pos = sum(1 for x in labels if x["sentiment"] == "positive")
            neg = sum(1 for x in labels if x["sentiment"] == "negative")
            return "positive" if pos > neg else "negative" if neg > pos else "neutral"
        except:
            return "neutral"

    reviews["_overall"] = reviews["label"].apply(overall)
    pos_reviews = reviews[reviews["_overall"] == "positive"].sort_values("_len", ascending=False)
    neg_reviews = reviews[reviews["_overall"] == "negative"].sort_values("_len", ascending=False)

    k     = get_top_k(len(reviews))
    pos_k = max(1, k - k // 3)   # 2/3 positive
    neg_k = max(1, k // 3)        # 1/3 negative

    picked = pd.concat([
        pos_reviews.head(pos_k),
        neg_reviews.head(neg_k),
    ]).drop_duplicates(subset=["content"]).head(k)

    docs = []
    for i, (_, r) in enumerate(picked.iterrows()):
        meta = build_metadata(det, scores)
        meta["review_rating"]  = safe_int(r["rating"])
        meta["review_overall"] = r["_overall"]
        docs.append({
            "id":       f"prod_{pid}_review_{i}",
            "doc_type": "review",
            "content":  (
                f"Đánh giá ({r['rating']}/5) về '{det.get('name', '')}':\n"
                f"{str(r['content'])[:600]}"
            ),
            "metadata": meta,
        })
    return docs


def build_metadata(det: pd.Series, scores: Optional[dict]) -> dict:
    meta = {
        "product_id":     int(det["product_id"]),
        "name":           det.get("name"),
        "brand_name":     det.get("brand_name"),
        "category_name":  det.get("category_name"),
        "price":          safe_int(det.get("price")),
        "rating_average": safe_float(det.get("rating_average")),
        "review_count":   safe_int(det.get("review_count")),
        "product_url":    det.get("product_url"),
    }
    if scores:
        meta["absa_n_reviews"] = safe_int(scores.get("n_reviews"))
        for asp in ASPECTS:
            meta[f"absa_{asp}_score"] = scores.get(f"{asp}_score")
            meta[f"absa_{asp}_pos"]   = safe_int(scores.get(f"{asp}_pos"))
            meta[f"absa_{asp}_neg"]   = safe_int(scores.get(f"{asp}_neg"))
            meta[f"absa_{asp}_conf"]  = scores.get(f"{asp}_conf")
    return {k: (None if isinstance(v, float) and v != v else v)
            for k, v in meta.items()}


# ── Main ───────────────────────────────────────────────
def main():
    print("[Stage 2] Loading inputs...")
    detail_df  = pd.read_csv(DETAIL_CSV)
    reviews_df = pd.read_csv(LABELED_CSV)
    scores_df  = pd.read_parquet(SCORES_PAR)
    print(f"  detail={len(detail_df):,}  reviews={len(reviews_df):,}  scored={len(scores_df):,}")

    scores_by_pid  = {int(r["product_id"]): r.to_dict() for _, r in scores_df.iterrows()}
    reviews_by_pid = {pid: grp for pid, grp in reviews_df.groupby("product_id")}

    stats = {"card": 0, "aspect": 0, "review": 0}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _, det in detail_df.iterrows():
            pid    = int(det["product_id"])
            scores = scores_by_pid.get(pid)
            revs   = reviews_by_pid.get(pid, pd.DataFrame())

            # 1) product_card
            f.write(json.dumps(build_product_card(det, scores), ensure_ascii=False) + "\n")
            stats["card"] += 1

            # 2) aspect_summary
            asp_doc = build_aspect_summary(det, scores)
            if asp_doc:
                f.write(json.dumps(asp_doc, ensure_ascii=False) + "\n")
                stats["aspect"] += 1

            # 3) reviews
            for rev_doc in build_reviews(det, revs, scores):
                f.write(json.dumps(rev_doc, ensure_ascii=False) + "\n")
                stats["review"] += 1

    total = sum(stats.values())
    print(f"\n[Stage 2] Done:")
    print(f"  product_card   : {stats['card']:,}")
    print(f"  aspect_summary : {stats['aspect']:,}")
    print(f"  review         : {stats['review']:,}")
    print(f"  Total          : {total:,} documents")
    print(f"  Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
