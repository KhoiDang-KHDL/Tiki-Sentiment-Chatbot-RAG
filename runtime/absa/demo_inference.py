"""Run from the project root: python -m runtime.absa.demo_inference --model logreg."""

import argparse
import json
from time import perf_counter

if __package__:
    from .inference import aggregate, get_logreg, get_phobert, get_phobert_onnx
else:
    from inference import aggregate, get_logreg, get_phobert, get_phobert_onnx


def main():
    parser = argparse.ArgumentParser(description="Run ABSA inference on sample reviews.")
    parser.add_argument(
        "--model", choices=["logreg", "phobert", "phobert_onnx"], default="logreg"
    )
    parser.add_argument(
        "--text", action="append", help="Review text; repeat this option for multiple reviews."
    )
    args = parser.parse_args()
    texts = args.text or [
        "Sản phẩm chất lượng tốt, đóng gói cẩn thận, giao hàng nhanh.",
        "Sản phẩm bị lỗi, giao hàng chậm, giá quá cao.",
        "Shop tư vấn nhiệt tình, sản phẩm đúng mô tả.",
    ]

    loaders = {
        "logreg": get_logreg,
        "phobert": get_phobert,
        "phobert_onnx": get_phobert_onnx,
    }
    print(f"Loading model: {args.model}", flush=True)
    start = perf_counter()
    predictor = loaders[args.model]()
    load_seconds = perf_counter() - start

    start = perf_counter()
    predictions = predictor.predict(texts)
    predict_seconds = perf_counter() - start

    print(json.dumps({
        "model": args.model,
        "reviews": [
            {"text": text, "sentiments": prediction}
            for text, prediction in zip(texts, predictions)
        ],
        "aggregate": aggregate(predictions),
        "load_seconds": round(load_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
