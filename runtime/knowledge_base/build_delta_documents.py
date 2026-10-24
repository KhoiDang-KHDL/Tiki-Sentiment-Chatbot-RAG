"""Rebuild affected products and retain direct incremental reviews across batches."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from runtime.preprocessing.clean_review import clean_text
from .build_documents import (
    build_product_card, build_aspect_summary, build_reviews, build_metadata, safe_int,
)
from .incremental_io import normalize_ids


def direct_review_id(pid, content):
    key = f"{int(pid)}|{clean_text(content)}"
    return f"prod_{int(pid)}_delta_{hashlib.sha256(key.encode()).hexdigest()}"


def validate_documents(docs):
    seen = set()
    for doc in docs:
        identifier = doc.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Document IDs must be non-empty and unique")
        seen.add(identifier)
        if not isinstance(doc.get("content"), str) or not doc["content"].strip():
            raise ValueError(f"Empty document: {identifier}")
        pid = doc.get("metadata", {}).get("product_id")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"Invalid metadata.product_id: {identifier}")
        if doc.get("doc_type") not in {"product_card", "aspect_summary", "review"}:
            raise ValueError(f"Invalid doc_type: {identifier}")
        json.dumps(doc, allow_nan=False)


def build_delta_documents(detail_df, reviews_df, scores_df, affected_ids,
                          delta_reviews=None, previous_docs=None):
    details, reviews, scores = map(normalize_ids, (detail_df, reviews_df, scores_df))
    affected = {int(pid) for pid in affected_ids}
    if details["product_id"].duplicated().any():
        raise ValueError("Duplicate product details")
    missing = affected - set(details["product_id"])
    if missing:
        raise ValueError(f"Missing product details: {sorted(missing)}")
    score_map = {int(row["product_id"]): row.to_dict() for _, row in scores.iterrows()}
    review_map = {int(pid): group for pid, group in reviews.groupby("product_id")}
    detail_map = {int(row["product_id"]): row for _, row in details.iterrows()}
    docs = []
    for pid in sorted(affected):
        detail, score = detail_map[pid], score_map.get(pid)
        docs.append(build_product_card(detail, score))
        summary = build_aspect_summary(detail, score)
        if summary:
            docs.append(summary)
        docs.extend(build_reviews(detail, review_map.get(pid, pd.DataFrame()), score))

    # Preserve direct reviews added by earlier incremental batches. Refresh their
    # ABSA metadata, rather than keeping stale product statistics in their payload.
    direct = {}
    for doc in previous_docs or []:
        pid = int(doc["metadata"]["product_id"])
        if pid in affected and doc["metadata"].get("incremental_review") is True:
            metadata = build_metadata(detail_map[pid], score_map.get(pid))
            for key in ("incremental_review", "review_rating", "review_overall"):
                if key in doc["metadata"]:
                    metadata[key] = doc["metadata"][key]
            direct[doc["id"]] = {**doc, "metadata": metadata}
    if delta_reviews is not None:
        for _, row in normalize_ids(delta_reviews).iterrows():
            pid = int(row["product_id"])
            if pid not in affected:
                raise ValueError("Delta review outside affected products")
            content = str(row["content"])
            identifier = direct_review_id(pid, content)
            metadata = build_metadata(detail_map[pid], score_map.get(pid))
            metadata.update(incremental_review=True, review_rating=safe_int(row["rating"]))
            direct[identifier] = {
                "id": identifier, "doc_type": "review",
                "content": f"Đánh giá ({row['rating']}/5) về '{detail_map[pid].get('name', '')}':\n{content[:600]}",
                "metadata": metadata,
            }
    docs.extend(direct.values())
    validate_documents(docs)
    return docs


def replace_product_documents(previous_docs, delta_docs, affected_ids):
    affected = {int(pid) for pid in affected_ids}
    result = [doc for doc in previous_docs
              if int(doc["metadata"]["product_id"]) not in affected] + list(delta_docs)
    validate_documents(result)
    return result
