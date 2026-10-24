from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
Stage 3-delta: Embed delta documents + update Qdrant incrementally.

Input:
- data/processed/delta_documents.jsonl

Logic:
- Không xóa toàn bộ collection.
- Chỉ xóa points cũ của affected product_id.
- Embed + upsert delta docs.
"""

import json
import uuid
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from FlagEmbedding import BGEM3FlagModel


BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_FILE = BASE_DIR / "data/processed/delta_documents.jsonl"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "tiki_kb")

EMBED_DIM = 1024
BATCH_SIZE = int(os.getenv("KB_INDEX_BATCH_SIZE", "16"))

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


def stable_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc_id)))


def load_documents() -> List[dict]:
    if not DOCS_FILE.exists():
        print(f"Không tìm thấy {DOCS_FILE}")
        return []

    docs = []
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    return docs


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' chưa tồn tại → tạo mới")

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
            print(f"Skip index {field}: {e}")


def get_affected_product_ids(docs: list[dict]) -> List[int]:
    ids = set()

    for doc in docs:
        meta = doc.get("metadata", {})
        pid = meta.get("product_id")
        if pid is not None:
            ids.add(int(pid))

    return sorted(ids)


def delete_old_points_for_products(client: QdrantClient, product_ids: list[int]):
    if not product_ids:
        return

    print(f"[Stage 3-delta] Deleting old Qdrant points for {len(product_ids)} products...")

    for pid in product_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="metadata.product_id",
                            match=qm.MatchValue(value=pid),
                        )
                    ]
                )
            ),
            wait=True,
        )

    print("[Stage 3-delta] Old affected-product points deleted")


def make_sparse_vector(svec: dict) -> qm.SparseVector:
    indices = []
    values = []

    for k, v in svec.items():
        try:
            indices.append(int(k))
            values.append(float(v))
        except Exception:
            continue

    return qm.SparseVector(indices=indices, values=values)


def embed_and_upsert(client: QdrantClient, model: BGEM3FlagModel, docs: list[dict]):
    total = len(docs)
    n_upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        texts = [str(d.get("content", "")) for d in batch]

        out = model.encode(
            texts,
            batch_size=len(batch),
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = out["dense_vecs"]
        sparse_vecs = out["lexical_weights"]

        points = []

        for doc, dvec, svec in zip(batch, dense_vecs, sparse_vecs):
            doc_id = doc["id"]

            points.append(
                qm.PointStruct(
                    id=stable_id(doc_id),
                    vector={
                        DENSE_NAME: dvec.tolist(),
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
        print(f"[Stage 3-delta] Upserted {n_upserted:,}/{total:,} delta docs", flush=True)

    return n_upserted


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("[Stage 3-delta] Loading delta documents...")
    docs = load_documents()
    print(f"  → {len(docs):,} delta documents")

    if not docs:
        print("[Stage 3-delta] Không có delta docs → skip indexing")
        return

    unique_ids = len({d["id"] for d in docs})
    if unique_ids != len(docs):
        raise RuntimeError(f"Delta doc id bị trùng: total={len(docs)}, unique={unique_ids}")

    product_ids = get_affected_product_ids(docs)
    print(f"  → {len(product_ids):,} affected products")

    print("[Stage 3-delta] Connecting to Qdrant")
    client = QdrantClient(url=QDRANT_URL, timeout=300)

    ensure_collection(client)

    before_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"  Qdrant count before: {before_count:,}")

    delete_old_points_for_products(client, product_ids)

    after_delete_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"  Qdrant count after delete: {after_delete_count:,}")

    print("[Stage 3-delta] Loading bge-m3 model")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    print(f"[Stage 3-delta] Embedding + upserting batch_size={BATCH_SIZE}")
    n_upserted = embed_and_upsert(client, model, docs)

    final_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print("[Stage 3-delta] Done")
    print(f"  Delta docs upserted : {n_upserted:,}")
    print(f"  Qdrant count before : {before_count:,}")
    print(f"  Qdrant count final  : {final_count:,}")


if __name__ == "__main__":
    main()
