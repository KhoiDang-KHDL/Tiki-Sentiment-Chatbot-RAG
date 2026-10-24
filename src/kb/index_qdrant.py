from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Stage 3: Embed documents với bge-m3 + upsert vào Qdrant

Input : data/processed/documents.jsonl
Output: Qdrant collection "tiki_kb"

CPU-safe version:
- Ép FlagEmbedding chạy CPU single-device.
- Không dùng CUDA để tránh torch CUDA OOM.
- Có log từng batch và verify final count.
"""

import os

# IMPORTANT: set trước khi import FlagEmbedding / torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("KB_INDEX_BATCH_SIZE", "8")
os.environ.setdefault("KB_MAX_LENGTH", "1024")

import gc
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from FlagEmbedding import BGEM3FlagModel


# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_FILE = BASE_DIR / "data/processed/documents.jsonl"

# ── Config ─────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "tiki_kb")

EMBED_DIM = 1024
BATCH_SIZE = int(os.getenv("KB_INDEX_BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("KB_MAX_LENGTH", "1024"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").lower().strip()

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


def stable_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc_id)))


def load_documents() -> List[dict]:
    docs = []
    if not DOCS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {DOCS_FILE}")

    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    return docs


def setup_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"  Collection '{COLLECTION_NAME}' đã tồn tại → xóa và tạo lại", flush=True)
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_NAME: qm.VectorParams(
                size=EMBED_DIM,
                distance=qm.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_NAME: qm.SparseVectorParams(),
        },
        on_disk_payload=True,
    )

    indexes = [
        ("doc_type", qm.PayloadSchemaType.KEYWORD),
        ("metadata.product_id", qm.PayloadSchemaType.INTEGER),
        ("metadata.price", qm.PayloadSchemaType.INTEGER),
        ("metadata.rating_average", qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_quality_score", qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_price_score", qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_delivery_score", qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_service_score", qm.PayloadSchemaType.FLOAT),
    ]

    for field, schema in indexes:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
        except Exception as e:
            # Nếu index đã tồn tại hoặc Qdrant báo duplicate thì bỏ qua.
            print(f"  Skip payload index {field}: {e}", flush=True)

    print(f"  Collection '{COLLECTION_NAME}' đã tạo xong", flush=True)


def make_sparse_vector(svec: dict) -> qm.SparseVector:
    indices = []
    values = []

    for k, v in (svec or {}).items():
        try:
            indices.append(int(k))
            values.append(float(v))
        except Exception:
            continue

    return qm.SparseVector(indices=indices, values=values)


def load_model() -> BGEM3FlagModel:
    """
    Ép CPU single-device để tránh:
    - CUDA out of memory
    - FlagEmbedding tự bật multi-process với target_devices rỗng
    """
    print(f"\n[Stage 3] Loading bge-m3 model on device={EMBEDDING_DEVICE}", flush=True)

    if EMBEDDING_DEVICE != "cpu":
        print("  WARN: bản này khuyến nghị EMBEDDING_DEVICE=cpu để tránh OOM", flush=True)

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False,
        devices=["cpu"] if EMBEDDING_DEVICE == "cpu" else [EMBEDDING_DEVICE],
    )

    # Guard cho một số version FlagEmbedding:
    # nếu target_devices bị [] thì encode sẽ đi vào encode_multi_process và lỗi.
    for attr in ("target_devices", "devices"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, ["cpu"] if EMBEDDING_DEVICE == "cpu" else [EMBEDDING_DEVICE])
            except Exception:
                pass

    if hasattr(model, "pool"):
        try:
            model.pool = None
        except Exception:
            pass

    return model


def embed_batch(model: BGEM3FlagModel, texts: list[str]) -> dict:
    """
    Gọi encode theo single-device path.
    """
    return model.encode(
        texts,
        batch_size=len(texts),
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def embed_and_upsert(client: QdrantClient, model: BGEM3FlagModel, docs: list[dict]) -> int:
    total = len(docs)
    n_upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        texts = [str(d.get("content", "")) for d in batch]

        print(f"[Stage 3] Embedding batch {i + 1:,}-{min(i + BATCH_SIZE, total):,}/{total:,}", flush=True)

        out = embed_batch(model, texts)

        dense_vecs = out["dense_vecs"]
        sparse_vecs = out.get("lexical_weights", [{} for _ in batch])

        points = []

        for doc, dvec, svec in zip(batch, dense_vecs, sparse_vecs):
            doc_id = doc["id"]

            points.append(
                qm.PointStruct(
                    id=stable_id(doc_id),
                    vector={
                        DENSE_NAME: dvec.tolist() if hasattr(dvec, "tolist") else dvec,
                        SPARSE_NAME: make_sparse_vector(svec),
                    },
                    payload={
                        "doc_id": doc_id,
                        "doc_type": doc.get("doc_type"),
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        n_upserted += len(points)

        count_now = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print(
            f"[Stage 3] Upserted {n_upserted:,}/{total:,} docs | Qdrant count={count_now:,}",
            flush=True,
        )

        gc.collect()

    return n_upserted


def main() -> None:
    print("[Stage 3] Loading documents...", flush=True)
    docs = load_documents()
    total = len(docs)
    unique_ids = len({d["id"] for d in docs})

    print(f"  → {total:,} documents", flush=True)
    print(f"  → {unique_ids:,} unique ids", flush=True)

    if total == 0:
        raise RuntimeError("documents.jsonl rỗng")

    if unique_ids != total:
        raise RuntimeError(f"ID bị trùng: total={total}, unique={unique_ids}")

    print("\n[Stage 3] Connecting to Qdrant", flush=True)
    print(f"  QDRANT_URL={QDRANT_URL}", flush=True)
    client = QdrantClient(url=QDRANT_URL, timeout=300)

    setup_collection(client)

    model = load_model()

    print(
        f"\n[Stage 3] Embedding + upserting batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}",
        flush=True,
    )
    n_upserted = embed_and_upsert(client, model, docs)

    final_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print("\n[Stage 3] Final check", flush=True)
    print(f"  Docs loaded     : {total:,}", flush=True)
    print(f"  Docs upserted   : {n_upserted:,}", flush=True)
    print(f"  Qdrant count    : {final_count:,}", flush=True)

    if final_count != total:
        raise RuntimeError(
            f"Qdrant count mismatch: expected {total:,}, got {final_count:,}"
        )

    print("\n[Stage 3] index_qdrant done", flush=True)


if __name__ == "__main__":
    main()
