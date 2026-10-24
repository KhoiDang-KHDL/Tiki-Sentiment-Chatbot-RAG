from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Stage 1: Aggregate review-level ABSA labels
→ product × aspect scores

Input : data/processed/labeled_processed_products_reviews.csv
Output: data/processed/product_aspect_scores.parquet
"""
import ast
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
LABELED_CSV = BASE_DIR / "data/processed/labeled_processed_products_reviews.csv"
OUTPUT_DIR  = BASE_DIR / "data/processed"
OUTPUT_FILE = OUTPUT_DIR / "product_aspect_scores.parquet"

# ── Constants ──────────────────────────────────────────
ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]
MIN_REVIEWS = 1  # cần ít nhất 1 review mới tính score


def parse_label(label_str: str) -> list:
    """Parse label column: "[{'aspect':..., 'sentiment':...}, ...]" """
    try:
        return ast.literal_eval(label_str)
    except Exception:
        return []


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Với mỗi (product_id × aspect), tính:
      - pos / neu / neg : số lượng reviews
      - score           : (pos - neg) / (pos + neg), range [-1, +1]
      - conf            : độ tin cậy (càng nhiều review càng cao)
    """
    bucket = defaultdict(lambda: defaultdict(list))
    review_counts = defaultdict(int)

    for _, row in df.iterrows():
        labels = parse_label(row["label"])
        if not labels:
            continue
        pid = int(row["product_id"])
        review_counts[pid] += 1
        for item in labels:
            asp = item.get("aspect")
            sen = item.get("sentiment")
            if asp in ASPECTS and sen in ("positive", "neutral", "negative"):
                bucket[pid][asp].append(sen)

    rows = []
    for pid, aspects_data in bucket.items():
        rec = {
            "product_id": pid,
            "n_reviews": review_counts[pid]
        }
        for asp in ASPECTS:
            sentiments = aspects_data.get(asp, [])
            pos = sum(1 for s in sentiments if s == "positive")
            neu = sum(1 for s in sentiments if s == "neutral")
            neg = sum(1 for s in sentiments if s == "negative")
            total = pos + neu + neg
            non_neutral = pos + neg

            # Score chỉ tính khi có đủ reviews
            if total < MIN_REVIEWS:
                score = None
            else:
                score = round((pos - neg) / non_neutral, 3) if non_neutral > 0 else 0.0

            # Confidence: bao nhiêu % của ngưỡng lý tưởng (MIN_REVIEWS × 3)
            conf = round(min(total / (MIN_REVIEWS * 3), 1.0), 3)

            rec[f"{asp}_pos"]   = pos
            rec[f"{asp}_neu"]   = neu
            rec[f"{asp}_neg"]   = neg
            rec[f"{asp}_score"] = score
            rec[f"{asp}_conf"]  = conf
        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    print(f"[Stage 1] Loading: {LABELED_CSV.name}")
    df = pd.read_csv(LABELED_CSV)
    print(f"  → {len(df):,} reviews, {df['product_id'].nunique()} products")

    print("[Stage 1] Aggregating ABSA labels...")
    scores_df = aggregate(df)
    print(f"  → {len(scores_df):,} products scored")

    # Sample output
    sample = scores_df.iloc[0]
    print(f"\n  Sample (product_id={sample['product_id']}):")
    for asp in ASPECTS:
        score = sample[f"{asp}_score"]
        pos   = int(sample[f"{asp}_pos"])
        neg   = int(sample[f"{asp}_neg"])
        print(f"    {asp:12s}: score={score}, pos={pos}, neg={neg}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n[Stage 1] Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
