"""Pure merge operation: existing historical rows are never deduplicated away."""
from __future__ import annotations

import pandas as pd

from runtime.preprocessing.clean_review import clean_text
from .incremental_io import normalize_ids, validate_labels


def review_keys(frame):
    if "content" not in frame:
        raise ValueError("Missing content column")
    return [(int(pid), clean_text(content)) for pid, content in
            zip(frame["product_id"], frame["content"])]


def merge_reviews(main_df, batch_df):
    main_df, batch_df = normalize_ids(main_df), normalize_ids(batch_df)
    validate_labels(main_df)
    validate_labels(batch_df)
    # Pending batch remains affected even if a previous attempt already merged it.
    affected = sorted(batch_df["product_id"].unique().tolist())
    existing = set(review_keys(main_df))
    keep = []
    for key in review_keys(batch_df):
        keep.append(bool(key[1]) and key not in existing)
        existing.add(key)
    added = batch_df.loc[keep].copy()
    merged = pd.concat([main_df, added], ignore_index=True, sort=False)
    return merged, added, affected
