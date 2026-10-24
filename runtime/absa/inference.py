from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
ABSA Inference — Load model từ local hoặc W&B, predict aspect sentiments
Supports: LogReg (fast, real-time) và PhoBERT (accurate, offline)
"""
import logging
from pathlib import Path
from runtime.preprocessing.clean_review import clean_text
from runtime.absa.paths import checkpoint_path

logger = logging.getLogger("absa.inference")

# Resolve local model paths independently of the working directory.
BASE_DIR   = Path(__file__).resolve().parents[2]
MODEL_DIR  = BASE_DIR / "models"
# The version argument is retained for API compatibility; local models are
# stored directly under models/<model_type>/ without version subdirectories.

ASPECTS    = ["description", "quality", "packaging", "delivery", "service", "price"]
LABEL_MAP  = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL  = {v: k for k, v in LABEL_MAP.items()}  # {0: "negative", 1: "neutral", 2: "positive"}
MODEL_NAME = "vinai/phobert-base"


# ─────────────────────────────────────────────────────────────────────────────
# LogReg (fast — dùng cho URL Analyzer real-time)
# ─────────────────────────────────────────────────────────────────────────────

class LogRegPredictor:
    def __init__(self, version: str = "v2"):
        import joblib
        model_path = checkpoint_path(BASE_DIR, "logreg", "logreg.pkl", version)
        if not model_path.exists():
            raise FileNotFoundError(f"LogReg model not found: {model_path}")
        bundle = joblib.load(model_path)
        self.vectorizer = bundle["vectorizer"]
        self.models     = bundle["models"]  # {aspect: LogisticRegression}
        self.aspects    = bundle["aspects"]
        logger.info(f"LogReg loaded from {model_path}")

    def predict(self, texts: list[str]) -> List[dict]:
        """
        Input:  ["review text 1", "review text 2", ...]
        Output: [
            {"description": "neutral", "quality": "positive", ...},
            ...
        ]
        """
        X = self.vectorizer.transform(texts)
        results = []
        aspect_preds = {asp: self.models[asp].predict(X) for asp in self.aspects}

        for i in range(len(texts)):
            row = {asp: INV_LABEL[aspect_preds[asp][i]] for asp in self.aspects}
            results.append(row)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# PhoBERT (accurate — dùng cho Airflow KB rebuild offline)
# ─────────────────────────────────────────────────────────────────────────────

class PhoBERTPredictor:
    def __init__(self, version: str = "v2"):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel

        model_path = checkpoint_path(BASE_DIR, "phobert", "phobert.pt", version)
        if not model_path.exists():
            raise FileNotFoundError(f"PhoBERT model not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint  = torch.load(model_path, map_location=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        self.model     = _MultiHeadPhoBERT(MODEL_NAME, hidden_size=768,
                                            num_classes=3, num_aspects=len(ASPECTS)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"PhoBERT loaded from {model_path} on {self.device}")

    def predict(self, texts: list[str], batch_size: int = 32) -> List[dict]:
        """
        Input:  ["review text 1", "review text 2", ...]
        Output: [{"description": "neutral", "quality": "positive", ...}, ...]
        """
        import torch
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)

            for j in range(len(batch_texts)):
                row = {}
                for k, asp in enumerate(ASPECTS):
                    pred_idx = int(torch.argmax(logits[k][j]).item())
                    row[asp] = INV_LABEL[pred_idx]
                all_results.append(row)

        return all_results


# ─────────────────────────────────────────────────────────────────────────────
# PhoBERT ONNX (optimized — dùng cho inference nhanh hơn)
# ─────────────────────────────────────────────────────────────────────────────

class PhoBERTONNXPredictor:
    def __init__(self, version: str = "v2"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_dir = checkpoint_path(BASE_DIR, "phobert_onnx", "phobert_absa.onnx", version).parent
        onnx_path = onnx_dir / "phobert_absa.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(str(onnx_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), use_fast=False)
        logger.info(f"PhoBERT ONNX loaded from {onnx_path}")

    def predict(self, texts: list[str], batch_size: int = 32) -> List[dict]:
        """
        Input:  ["review text 1", "review text 2", ...]
        Output: [{"description": "neutral", "quality": "positive", ...}, ...]
        """
        import numpy as np
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="np",
            )

            outputs = self.session.run(None, {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })

            logits = outputs[0]  # [batch, 6, 3]

            for j in range(len(batch_texts)):
                row = {}
                for k, asp in enumerate(ASPECTS):
                    pred_idx = int(np.argmax(logits[j][k]))
                    row[asp] = INV_LABEL[pred_idx]
                all_results.append(row)

        return all_results


class _MultiHeadPhoBERT:
    """Định nghĩa lại architecture để load được state_dict."""
    def __new__(cls, model_name, hidden_size, num_classes, num_aspects, dropout=0.3):
        import torch.nn as nn
        from transformers import AutoModel

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.phobert  = AutoModel.from_pretrained(model_name)
                self.dropout  = nn.Dropout(dropout)
                self.heads    = nn.ModuleList([
                    nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)
                ])

            def forward(self, input_ids, attention_mask):
                out     = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
                cls_out = self.dropout(out.last_hidden_state[:, 0])
                return [head(cls_out) for head in self.heads]

        return _Model()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate: labels → ABSA scores (giống aggregate_absa.py)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(predictions: list[dict]) -> dict:
    """
    Input:  list of per-review predictions từ predict()
    Output: {
        "quality": {"score": 0.85, "pos": 34, "neg": 3, "pct": 92, "total": 37},
        ...
    }
    """
    counts = {asp: {"pos": 0, "neg": 0} for asp in ASPECTS}

    for pred in predictions:
        for asp in ASPECTS:
            sentiment = pred.get(asp, "neutral")
            if sentiment == "positive":
                counts[asp]["pos"] += 1
            elif sentiment == "negative":
                counts[asp]["neg"] += 1

    scores = {}
    for asp in ASPECTS:
        pos   = counts[asp]["pos"]
        neg   = counts[asp]["neg"]
        total = pos + neg
        score = round((pos - neg) / total, 3) if total > 0 else 0.0
        pct   = round(pos / total * 100) if total > 0 else 0
        scores[asp] = {"score": score, "pos": pos, "neg": neg, "pct": pct, "total": total}

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Public API — dùng trong url_analyzer.py
# ─────────────────────────────────────────────────────────────────────────────

_logreg_predictor        = None
_phobert_predictor       = None
_phobert_onnx_predictor  = None


def get_logreg(version: str = "v2") -> LogRegPredictor:
    global _logreg_predictor
    if _logreg_predictor is None:
        _logreg_predictor = LogRegPredictor(version)
    return _logreg_predictor


def get_phobert(version: str = "v2") -> PhoBERTPredictor:
    global _phobert_predictor
    if _phobert_predictor is None:
        _phobert_predictor = PhoBERTPredictor(version)
    return _phobert_predictor


def get_phobert_onnx(version: str = "v2") -> PhoBERTONNXPredictor:
    global _phobert_onnx_predictor
    if _phobert_onnx_predictor is None:
        _phobert_onnx_predictor = PhoBERTONNXPredictor(version)
    return _phobert_onnx_predictor


def predict_and_aggregate(reviews: list[dict], model: str = "logreg", version: str = "v2") -> dict:
    """
    Convenience function — nhận reviews list, trả ABSA scores.
    Dùng trong url_analyzer.py thay cho _mock_absa_inference().

    Args:
        reviews: [{"content": "text", "rating": 5}, ...]
        model:   "logreg" (fast) hoặc "phobert" (accurate) hoặc "phobert_onnx"
        version: Giữ để tương thích API; không dùng trong đường dẫn local.

    Returns:
        {"quality": {"score": 0.85, "pos": 34, ...}, ...}
    """
    # Training samples use clean_content. Apply the same normalization for all
    # inference callers; never refit the saved vectorizer on newly crawled data.
    texts = [clean_text(r.get("clean_content") if isinstance(r.get("clean_content"), str)
                        else r.get("content", "")) for r in reviews]
    texts = [text for text in texts if text]
    if not texts:
        return {asp: {"score": 0.0, "pos": 0, "neg": 0, "pct": 0, "total": 0} for asp in ASPECTS}

    if model == "logreg":
        predictor = get_logreg(version)
    elif model == "phobert_onnx":
        predictor = get_phobert_onnx(version)
    else:
        predictor = get_phobert(version)

    predictions = predictor.predict(texts)
    return aggregate(predictions)
