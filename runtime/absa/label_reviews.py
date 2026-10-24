"""Clean and label a new CSV without changing the main review dataset."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from runtime.preprocessing.clean_review import clean_review
from runtime.knowledge_base.incremental_io import (
    ASPECTS, SENTIMENTS, normalize_ids, read_csv, validate_labels, write_csv,
)


def prepare_reviews(frame):
    missing = {"product_id", "content", "rating"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing review columns: {sorted(missing)}")
    frame = normalize_ids(frame)
    frame["content"] = frame["content"].fillna("").astype(str)
    ratings = pd.to_numeric(frame["rating"], errors="raise")
    if not (ratings.between(1, 5) & (ratings % 1 == 0)).all():
        raise ValueError("rating must be an integer from 1 to 5")
    frame["rating"] = ratings.astype(int)
    return clean_review(frame).reset_index(drop=True)


def label_frame(frame, model="logreg", predictor=None):
    frame = prepare_reviews(frame)
    if frame.empty:
        frame["label"] = pd.Series(dtype=str)
        return frame
    if predictor is None:
        from runtime.absa.inference import get_logreg, get_phobert, get_phobert_onnx
        loaders = {"logreg": get_logreg, "phobert": get_phobert,
                   "phobert_onnx": get_phobert_onnx}
        if model not in loaders:
            raise ValueError(f"Unsupported model: {model}")
        predictor = loaders[model]()
    predictions = predictor.predict(frame["clean_content"].tolist())
    if len(predictions) != len(frame):
        raise ValueError("Prediction count differs from review count")
    labels = []
    for prediction in predictions:
        if any(prediction.get(aspect) not in SENTIMENTS for aspect in ASPECTS):
            raise ValueError("Predictor returned invalid labels")
        labels.append(json.dumps([
            {"aspect": aspect, "sentiment": prediction[aspect]} for aspect in ASPECTS
        ], ensure_ascii=False))
    frame["label"] = labels
    validate_labels(frame)
    return frame


def label_reviews(input_path, model="logreg"):
    return label_frame(read_csv(input_path), model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["logreg", "phobert", "phobert_onnx"], default="logreg")
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.output).exists():
        parser.error("Output exists; choose a new output path")
    result = label_reviews(args.input, args.model)
    write_csv(result, args.output)
    print(f"Labeled {len(result)} reviews -> {args.output}")


if __name__ == "__main__":
    main()
