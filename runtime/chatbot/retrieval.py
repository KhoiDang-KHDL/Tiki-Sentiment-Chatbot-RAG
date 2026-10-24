"""
Dense retrieval for the BGE-M3 collection created by index_qdrant.py.
Run: python -m runtime.chatbot.retrieval --query "sản phẩm chất lượng tốt"
Redis is optional and is enabled only when REDIS_URL is configured.
"""
from __future__ import annotations
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import os
import hashlib
import json
import numpy as np

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = os.getenv("QDRANT_COLLECTION", "tiki_kb")
EMBED_MODEL_NAME = "BAAI/bge-m3"
ASPECTS          = ["description", "quality", "packaging", "delivery", "service", "price"]

ONNX_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-m3-onnx"
EMBED_DIM = 1024
MAX_LENGTH = int(os.getenv("KB_QUERY_MAX_LENGTH", "256"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _validate_dense(value) -> list[float]:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (EMBED_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite {EMBED_DIM}-dimensional embedding, got {vector.shape}")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite, non-zero norm")
    return (vector / norm).tolist()


@dataclass
class RagFilters:
    doc_types: list[str] = field(default_factory=list)
    min_rating: Optional[float] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    aspect_preferences: dict[str, float] = field(default_factory=dict)


def _build_qdrant_filter(f: RagFilters):
    from qdrant_client.http import models as qm
    must = []
    if f.price_min is not None and f.price_max is not None and f.price_min > f.price_max:
        raise ValueError("price_min must not exceed price_max")
    unknown = set(f.aspect_preferences) - set(ASPECTS)
    if unknown:
        raise ValueError(f"Unknown aspects: {', '.join(sorted(unknown))}")
    if f.doc_types:
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=f.doc_types)))
    if f.min_rating is not None:
        must.append(qm.FieldCondition(key="metadata.rating_average", range=qm.Range(gte=f.min_rating)))
    if f.price_min is not None or f.price_max is not None:
        must.append(qm.FieldCondition(key="metadata.price", range=qm.Range(gte=f.price_min, lte=f.price_max)))
    for aspect, min_score in f.aspect_preferences.items():
        if not np.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("Aspect thresholds must be finite values between -1 and 1")
        must.append(qm.FieldCondition(key=f"metadata.absa_{aspect}_score", range=qm.Range(gte=min_score)))
    return qm.Filter(must=must) if must else None


class TikiRAG:
    def __init__(self):
        from qdrant_client import QdrantClient

        if MAX_LENGTH <= 0:
            raise ValueError("KB_QUERY_MAX_LENGTH must be positive")
        self.client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.close()
            raise RuntimeError(f"Collection {COLLECTION_NAME!r} does not exist. Run index_qdrant first using the same QDRANT_COLLECTION.")
        vectors = self.client.get_collection(COLLECTION_NAME).config.params.vectors
        if not isinstance(vectors, dict) or "dense" not in vectors or vectors["dense"].size != EMBED_DIM:
            self.client.close()
            raise RuntimeError("Collection must contain a named 'dense' vector of size 1024")
        self.cache = None
        if os.getenv("REDIS_URL"):
            try:
                import redis
                self.cache = redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=1, socket_timeout=1)
            except ImportError:
                print("TikiRAG: redis is not installed; cache disabled")
        self.onnx_session = None
        self.tokenizer = None
        self.flag_model = None

        onnx_path = ONNX_DIR / "bge_m3_dense.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                self.onnx_session = ort.InferenceSession(str(onnx_path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
                print("TikiRAG: Using ONNX dense encoder (fast mode)")
            except Exception as e:
                self.onnx_session = None
                self.tokenizer = None
                print(f"TikiRAG: ONNX load failed ({e}), falling back to FlagEmbedding")

        if self.onnx_session is None:
            from FlagEmbedding import BGEM3FlagModel
            self.flag_model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=False, devices=[EMBEDDING_DEVICE])
            print("TikiRAG: Using FlagEmbedding (full mode)")
        encoder = "flag-fp32" if self.onnx_session is None else f"onnx:{onnx_path.stat().st_mtime_ns}:{onnx_path.stat().st_size}"
        self.cache_namespace = f"v2:{EMBED_MODEL_NAME}:{encoder}:{MAX_LENGTH}"

    def _encode_onnx(self, text: str) -> List[float]:
        enc = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        feed = {}
        for item in self.onnx_session.get_inputs():
            if item.name not in enc:
                raise ValueError(f"Tokenizer did not provide ONNX input: {item.name}")
            feed[item.name] = enc[item.name].astype(np.int64)
        output = np.asarray(self.onnx_session.run(None, feed)[0])
        if output.ndim == 3 and output.shape[0] == 1:
            dense = output[0, 0]
        elif output.ndim == 2 and output.shape[0] == 1:
            dense = output[0]
        else:
            raise ValueError(f"Unsupported ONNX output shape: {output.shape}")
        return _validate_dense(dense)

    def search(self, query: str, top_k: int = 5, filters: RagFilters | None = None, absa_rerank_weight: float = 0.15) -> List[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not np.isfinite(absa_rerank_weight) or absa_rerank_weight < 0:
            raise ValueError("absa_rerank_weight must be finite and non-negative")
        query = query.strip()
        qfilter = _build_qdrant_filter(filters) if filters else None
        cache_key = "emb:" + hashlib.sha256(f"{self.cache_namespace}:{query}".encode()).hexdigest()
        dense = None
        if self.cache is not None:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    dense = _validate_dense(json.loads(cached))
            except Exception:
                pass

        if dense is None:
            if self.onnx_session is not None:
                dense = self._encode_onnx(query)
            else:
                out = self.flag_model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False, max_length=MAX_LENGTH)
                dense = _validate_dense(out["dense_vecs"][0])
            if self.cache is not None:
                try:
                    self.cache.set(cache_key, json.dumps(dense), ex=3600)
                except Exception:
                    pass

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense.tolist() if hasattr(dense, 'tolist') else list(dense),
            using="dense",
            limit=top_k * 2,
            query_filter=qfilter,
            with_payload=True,
        ).points

        docs = []
        for pt in results:
            p = pt.payload or {}
            metadata = p.get("metadata") or {}
            absa_boost = 0.0
            if filters and filters.aspect_preferences:
                for asp in filters.aspect_preferences:
                    value = metadata.get(f"absa_{asp}_score")
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        absa_boost += value
                absa_boost /= len(filters.aspect_preferences)
            docs.append({
                "id":       p.get("doc_id", pt.id),
                "score":    pt.score + absa_rerank_weight * absa_boost,
                "doc_type": p.get("doc_type"),
                "text":     p.get("content", ""),
                "metadata": metadata,
            })

        docs.sort(key=lambda x: x["score"], reverse=True)
        return docs[:top_k]

    def close(self):
        self.client.close()
        if self.cache is not None:
            self.cache.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search the indexed Tiki documents")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not args.query.strip() or args.top_k <= 0:
        parser.error("query must not be blank and top-k must be positive")
    rag = TikiRAG()
    try:
        print(json.dumps(rag.search(args.query, args.top_k), ensure_ascii=False, indent=2))
    finally:
        rag.close()
