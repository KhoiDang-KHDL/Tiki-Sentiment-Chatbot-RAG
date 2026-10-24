"""Upsert affected product documents, then remove stale IDs. Never recreate a collection."""
from __future__ import annotations

import math
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .build_delta_documents import validate_documents


def stable_id(doc_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc_id)))


def product_filter(pid):
    return qm.Filter(must=[qm.FieldCondition(
        key="metadata.product_id", match=qm.MatchValue(value=int(pid)))])


def product_points(client, collection, pid):
    records, offset = {}, None
    while True:
        points, offset = client.scroll(
            collection_name=collection, scroll_filter=product_filter(pid),
            limit=256, offset=offset, with_payload=True, with_vectors=False)
        records.update((str(point.id), point.payload) for point in points)
        if offset is None:
            return records


def check_collection(client, collection):
    if not client.collection_exists(collection):
        raise ValueError(f"Collection {collection!r} does not exist; initialize it first")
    params = client.get_collection(collection).config.params
    vectors = params.vectors
    if (not isinstance(vectors, dict) or "dense" not in vectors
            or vectors["dense"].size != 1024
            or vectors["dense"].distance != qm.Distance.COSINE
            or "sparse" not in (params.sparse_vectors or {})):
        raise ValueError("Collection must use BGE-M3 dense(1024, Cosine) and sparse vectors")


def update_qdrant(docs, *, collection, url="http://localhost:6333", client=None,
                  model=None, batch_size=2, max_length=1024):
    validate_documents(docs)
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    if not docs:
        return {"documents": 0, "products": 0, "deleted": 0}
    own_client = client is None
    client = client or QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=300)
    try:
        check_collection(client, collection)
        expected = {}
        for doc in docs:
            pid = doc["metadata"]["product_id"]
            expected.setdefault(pid, {})[stable_id(doc["id"])] = {
                "doc_id": doc["id"], "doc_type": doc["doc_type"],
                "content": doc["content"], "metadata": doc["metadata"],
            }
        old = {pid: product_points(client, collection, pid) for pid in expected}
        if model is None:
            from .index_qdrant import load_model
            model = load_model()
        # Finish every upsert before deleting anything. If embedding/upsert fails,
        # retry the pending batch; stable IDs overwrite any partially written data.
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            output = model.encode([doc["content"] for doc in batch], batch_size=len(batch),
                                  max_length=max_length, return_dense=True,
                                  return_sparse=True, return_colbert_vecs=False)
            dense, sparse = output["dense_vecs"], output["lexical_weights"]
            if len(dense) != len(batch) or len(sparse) != len(batch):
                raise ValueError("Embedding count differs from document count")
            from .index_qdrant import make_sparse_vector
            points = []
            for doc, vector, lexical in zip(batch, dense, sparse):
                vector = [float(value) for value in vector]
                if len(vector) != 1024 or not all(math.isfinite(value) for value in vector) or not any(vector):
                    raise ValueError(f"Invalid dense embedding: {doc['id']}")
                pid, identifier = doc["metadata"]["product_id"], stable_id(doc["id"])
                points.append(qm.PointStruct(id=identifier, vector={
                    "dense": vector, "sparse": make_sparse_vector(lexical)},
                    payload=expected[pid][identifier]))
            client.upsert(collection_name=collection, points=points, wait=True)
            print(f"Upserted {min(start + batch_size, len(docs))}/{len(docs)} documents", flush=True)
        deleted = 0
        for pid, desired in expected.items():
            stale = sorted(set(old[pid]) - set(desired))
            for start in range(0, len(stale), 256):
                client.delete(collection_name=collection,
                              points_selector=qm.PointIdsList(points=stale[start:start + 256]), wait=True)
            deleted += len(stale)
        for pid, desired in expected.items():
            if product_points(client, collection, pid) != desired:
                raise RuntimeError(f"Verification failed for product {pid}: IDs or payloads differ")
        return {"documents": len(docs), "products": len(expected), "deleted": deleted}
    finally:
        if own_client:
            client.close()
