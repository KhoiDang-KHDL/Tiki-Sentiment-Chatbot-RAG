from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Build delta documents cho Qdrant.

Logic:
- Rebuild product_card + aspect_summary cho affected products.
- Rebuild top-k review docs như cũ.
- Thêm trực tiếp tất cả review mới từ kb_delta_reviews.csv vào KB,
  để review mới chắc chắn được đưa vào Qdrant, không phụ thuộc top_k.

Input:
- data/processed/kb_delta_product_ids.txt
- data/processed/kb_delta_reviews.csv
- data/raw/products_detail.csv
- data/processed/labeled_processed_products_reviews.csv
- data/processed/product_aspect_scores.parquet

Output:
- data/processed/delta_documents.jsonl
"""

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.kb.build_documents import (
    build_product_card,
    build_aspect_summary,
    build_reviews,
    build_metadata,
    safe_int,
)


BASE_DIR = Path(__file__).resolve().parents[2]

DETAIL_CSV = BASE_DIR / "data/raw/products_detail.csv"
LABELED_CSV = BASE_DIR / "data/processed/labeled_processed_products_reviews.csv"
SCORES_PAR = BASE_DIR / "data/processed/product_aspect_scores.parquet"

AFFECTED_IDS_FILE = BASE_DIR / "data/processed/kb_delta_product_ids.txt"
DELTA_REVIEWS_CSV = BASE_DIR / "data/processed/kb_delta_reviews.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/delta_documents.jsonl"


def load_affected_product_ids() -> set[int]:
    if not AFFECTED_IDS_FILE.exists():
        print(f"Không tìm thấy {AFFECTED_IDS_FILE}")
        return set()

    ids = []
    for line in AFFECTED_IDS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(int(line))

    return set(ids)


def load_delta_reviews() -> pd.DataFrame:
    if not DELTA_REVIEWS_CSV.exists():
        print(f"Không tìm thấy {DELTA_REVIEWS_CSV} → không có direct delta reviews")
        return pd.DataFrame()

    df = pd.read_csv(DELTA_REVIEWS_CSV)

    required_cols = {"product_id", "content"}
    if not required_cols.issubset(df.columns):
        print(f"Delta reviews thiếu cột {required_cols - set(df.columns)} → bỏ qua direct reviews")
        return pd.DataFrame()

    df["product_id"] = df["product_id"].astype(int)
    df["content"] = df["content"].astype(str)

    df = df.drop_duplicates(subset=["product_id", "content"]).copy()

    return df


def hash_review_id(product_id: int, content: str) -> str:
    raw = f"{product_id}|{content}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"prod_{product_id}_delta_review_{h}"


def infer_overall_from_label(label_str) -> str:
    try:
        labels = ast.literal_eval(label_str) if isinstance(label_str, str) else []
        pos = sum(1 for x in labels if x.get("sentiment") == "positive")
        neg = sum(1 for x in labels if x.get("sentiment") == "negative")

        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
    except Exception:
        return "neutral"


def build_direct_delta_review_docs(
    det: pd.Series,
    delta_reviews: pd.DataFrame,
    scores: Optional[dict],
) -> List[dict]:
    """
    Build docs cho tất cả review mới.
    Các docs này không phụ thuộc top_k.
    """
    pid = int(det["product_id"])

    if delta_reviews.empty:
        return []

    product_delta = delta_reviews[delta_reviews["product_id"].astype(int) == pid].copy()

    if product_delta.empty:
        return []

    docs = []

    for _, r in product_delta.iterrows():
        content = str(r.get("content", "")).strip()

        if not content:
            continue

        meta = build_metadata(det, scores)

        if "rating" in r:
            meta["review_rating"] = safe_int(r.get("rating"))

        if "label" in r:
            meta["review_overall"] = infer_overall_from_label(r.get("label"))

        meta["is_new_review"] = True
        meta["source"] = "delta_review"

        source_file = r.get("_source_file")
        if source_file is not None and str(source_file) != "nan":
            meta["source_file"] = str(source_file)

        doc_id = hash_review_id(pid, content)

        docs.append(
            {
                "id": doc_id,
                "doc_type": "new_review",
                "content": (
                    f"Review mới về '{det.get('name', '')}':\n"
                    f"{content[:800]}"
                ),
                "metadata": meta,
            }
        )

    return docs


def main():
    print("[Stage 2-delta] Loading affected product ids...")
    affected_ids = load_affected_product_ids()

    if not affected_ids:
        print("[Stage 2-delta] Không có affected product ids → không build delta docs")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text("", encoding="utf-8")
        return

    print(f"  → {len(affected_ids):,} affected products")

    print("[Stage 2-delta] Loading delta reviews...")
    delta_reviews_df = load_delta_reviews()
    print(f"  → {len(delta_reviews_df):,} direct delta reviews")

    print("[Stage 2-delta] Loading inputs...")
    detail_df = pd.read_csv(DETAIL_CSV)
    reviews_df = pd.read_csv(LABELED_CSV)
    scores_df = pd.read_parquet(SCORES_PAR)

    detail_df["product_id"] = detail_df["product_id"].astype(int)
    reviews_df["product_id"] = reviews_df["product_id"].astype(int)
    scores_df["product_id"] = scores_df["product_id"].astype(int)

    detail_df = detail_df[detail_df["product_id"].isin(affected_ids)].copy()

    scores_by_pid = {
        int(r["product_id"]): r.to_dict()
        for _, r in scores_df[scores_df["product_id"].isin(affected_ids)].iterrows()
    }

    reviews_by_pid = {
        int(pid): grp
        for pid, grp in reviews_df[reviews_df["product_id"].isin(affected_ids)].groupby("product_id")
    }

    stats = {
        "card": 0,
        "aspect": 0,
        "top_review": 0,
        "new_review": 0,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_doc_ids = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _, det in detail_df.iterrows():
            pid = int(det["product_id"])
            scores = scores_by_pid.get(pid)
            revs = reviews_by_pid.get(pid, pd.DataFrame())

            # 1. product_card
            card_doc = build_product_card(det, scores)
            if card_doc["id"] not in seen_doc_ids:
                f.write(json.dumps(card_doc, ensure_ascii=False) + "\n")
                seen_doc_ids.add(card_doc["id"])
                stats["card"] += 1

            # 2. aspect_summary
            asp_doc = build_aspect_summary(det, scores)
            if asp_doc and asp_doc["id"] not in seen_doc_ids:
                f.write(json.dumps(asp_doc, ensure_ascii=False) + "\n")
                seen_doc_ids.add(asp_doc["id"])
                stats["aspect"] += 1

            # 3. top-k review docs như logic cũ
            for rev_doc in build_reviews(det, revs, scores):
                if rev_doc["id"] not in seen_doc_ids:
                    f.write(json.dumps(rev_doc, ensure_ascii=False) + "\n")
                    seen_doc_ids.add(rev_doc["id"])
                    stats["top_review"] += 1

            # 4. direct new review docs: luôn thêm review mới vào KB
            for new_doc in build_direct_delta_review_docs(det, delta_reviews_df, scores):
                if new_doc["id"] not in seen_doc_ids:
                    f.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
                    seen_doc_ids.add(new_doc["id"])
                    stats["new_review"] += 1

    total = sum(stats.values())

    print("[Stage 2-delta] Done:")
    print(f"  product_card   : {stats['card']:,}")
    print(f"  aspect_summary : {stats['aspect']:,}")
    print(f"  top_review     : {stats['top_review']:,}")
    print(f"  new_review     : {stats['new_review']:,}")
    print(f"  Total          : {total:,} delta documents")
    print(f"  Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
