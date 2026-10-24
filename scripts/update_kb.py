"""Local, resumable incremental KB updates. Run with python -m scripts.update_kb."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from runtime.absa.label_reviews import label_frame, prepare_reviews
from runtime.knowledge_base.aggregate_absa import aggregate
from runtime.knowledge_base.build_delta_documents import (
    build_delta_documents, replace_product_documents, validate_documents,
)
from runtime.knowledge_base.incremental_io import (
    atomic_write, batch_lock, fingerprint, normalize_ids, read_csv, read_jsonl,
    write_csv, write_json, write_jsonl,
)
from runtime.knowledge_base.merge_reviews import merge_reviews

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "incremental-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def model_revision(root, model):
    relative = {"logreg": "logreg/logreg.pkl", "phobert": "phobert/phobert.pt",
                "phobert_onnx": "phobert_onnx/phobert_absa.onnx"}[model]
    from runtime.absa.paths import checkpoint_path
    path = checkpoint_path(root, model, Path(relative).name)
    relative = str(path.relative_to(root / "models"))
    # Identity prevents reuse of a completed batch after checkpoint replacement.
    # Use a new --revision if auxiliary tokenizer files or model code change.
    if not path.exists():
        return {"path": relative, "missing": True}
    stat = path.stat()
    return {"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run(input_path, *, model="logreg", collection=None, url=None, root=PROJECT_ROOT,
        dry_run=False, revision="1", batch_size=None, max_length=None,
        labeler=None, updater=None):
    root, input_path = Path(root).resolve(), Path(input_path).resolve()
    if model not in {"logreg", "phobert", "phobert_onnx"}:
        raise ValueError(f"Unsupported model: {model}")
    collection = collection or os.getenv("QDRANT_COLLECTION")
    if not collection or not collection.strip():
        raise ValueError("Set QDRANT_COLLECTION or pass --collection explicitly")
    collection = collection.strip()
    url = (url or os.getenv("QDRANT_URL", "http://localhost:6333")).rstrip("/")
    batch_size = int(batch_size if batch_size is not None else os.getenv("KB_INDEX_BATCH_SIZE", "2"))
    max_length = int(max_length if max_length is not None else os.getenv("KB_MAX_LENGTH", "1024"))
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("Embedding batch size and max length must be positive")
    processed = root / "data" / "processed"
    paths = {
        "main": processed / "labeled_processed_products_reviews.csv",
        "scores": processed / "product_aspect_scores.parquet",
        "documents": processed / "documents.jsonl",
    }
    details_path = root / "data" / "raw" / "products_detail.csv"
    for path in [input_path, details_path, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_digest = fingerprint(input_path)
    raw = read_csv(input_path)
    cleaned = prepare_reviews(raw)
    details = normalize_ids(read_csv(details_path))
    if details["product_id"].duplicated().any():
        raise ValueError("Duplicate product details")
    missing = set(cleaned["product_id"]) - set(details["product_id"])
    if missing:
        raise ValueError(f"Missing product details: {sorted(missing)}. Add details before updating KB.")
    identity = {
        "input_sha256": input_digest, "model": model, "model_revision": model_revision(root, model),
        "pipeline": PIPELINE_VERSION, "revision": str(revision),
        "url": url, "collection": collection, "embedding": "BAAI/bge-m3",
        "max_length": max_length,
    }
    batch_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    batch_root = processed / "kb_batches"
    job = batch_root / batch_id
    report = {"batch_id": batch_id, "collection": collection, "input_rows": len(raw),
              "clean_rows": len(cleaned), "filtered_rows": len(raw) - len(cleaned),
              "affected_products": sorted(cleaned["product_id"].unique().tolist())}
    if dry_run:
        # No model loads, no Qdrant requests, and no filesystem mutations.
        existing = normalize_ids(read_csv(paths["main"]))
        from runtime.knowledge_base.merge_reviews import review_keys
        known = set(review_keys(existing))
        report.update(dry_run=True, new_reviews=sum(key not in known for key in review_keys(cleaned)))
        return report

    with batch_lock(batch_root / ".update.lock"):
        pending = batch_root / "pending.json"
        if pending.exists():
            active = json.loads(pending.read_text(encoding="utf-8"))
            if active["batch_id"] != batch_id:
                raise RuntimeError(
                    f"Batch {active['batch_id']} is pending. Retry its original input/model/target "
                    "before starting another batch. See kb_batches/pending.json.")
        if (job / "done.json").exists():
            result = json.loads((job / "done.json").read_text(encoding="utf-8"))
            if pending.exists():
                pending.unlink()
            return {**result, "already_done": True}
        if updater is None and not cleaned.empty:
            # Reject wrong targets/outages before inference and canonical writes.
            from qdrant_client import QdrantClient
            from runtime.knowledge_base.index_qdrant_delta import check_collection
            client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
            try:
                check_collection(client, collection)
            finally:
                client.close()
        job.mkdir(parents=True, exist_ok=True)
        manifest_path = job / "manifest.json"
        if not manifest_path.exists():
            original_hashes = {key: fingerprint(path) for key, path in paths.items()}
            details_hash = fingerprint(details_path)
            previous_docs = read_jsonl(paths["documents"])
            validate_documents(previous_docs)
            if fingerprint(input_path) != input_digest:
                raise RuntimeError("Input changed while being read; retry")
            labeled_path = job / "labeled_reviews.csv"
            if labeled_path.exists():
                labeled = read_csv(labeled_path)
            else:
                labeled = (labeler or label_frame)(cleaned, model=model)
                write_csv(labeled, labeled_path)
            merged, added, affected = merge_reviews(read_csv(paths["main"]), labeled)
            scores = aggregate(merged)
            delta = build_delta_documents(details, merged, scores, affected,
                                          delta_reviews=labeled, previous_docs=previous_docs)
            documents = replace_product_documents(previous_docs, delta, affected)
            write_csv(merged, job / "main.csv")
            atomic_write(job / "scores.parquet", lambda temp: scores.to_parquet(temp, index=False))
            write_jsonl(documents, job / "documents.jsonl")
            write_jsonl(delta, job / "delta_documents.jsonl")
            write_csv(labeled, job / "kb_delta_reviews.csv")
            atomic_write(job / "kb_delta_product_ids.txt", lambda temp: temp.write_text(
                "\n".join(map(str, affected)), encoding="utf-8"))
            # Preserve originals before any canonical files are changed.
            for key, path in paths.items():
                atomic_write(job / ("before_" + path.name), lambda temp, source=path: shutil.copyfile(source, temp))
            if original_hashes != {key: fingerprint(path) for key, path in paths.items()} or fingerprint(details_path) != details_hash:
                raise RuntimeError("Source data changed during preparation; retry without concurrent writers")
            snapshots = {"main": "main.csv", "scores": "scores.parquet", "documents": "documents.jsonl"}
            artifact_names = [*snapshots.values(), "delta_documents.jsonl", "labeled_reviews.csv",
                              "kb_delta_reviews.csv", "kb_delta_product_ids.txt"]
            manifest = {
                **report, "identity": identity, "input": str(input_path), "created_at": now(),
                "state": "prepared", "new_reviews": len(added), "affected_products": affected,
                "original_hashes": original_hashes, "details_hash": details_hash,
                "snapshots": snapshots,
                "artifact_hashes": {name: fingerprint(job / name) for name in artifact_names},
            }
            write_json(manifest, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise RuntimeError("Batch configuration differs from prepared manifest")
        if fingerprint(details_path) != manifest["details_hash"]:
            raise RuntimeError("Product details changed during a pending batch; restore them before retry")
        for name, expected_hash in manifest["artifact_hashes"].items():
            if fingerprint(job / name) != expected_hash:
                raise RuntimeError(f"Prepared artifact changed or missing: {name}")
        for key, path in paths.items():
            allowed = {manifest["original_hashes"][key],
                       manifest["artifact_hashes"][manifest["snapshots"][key]]}
            if fingerprint(path) not in allowed:
                raise RuntimeError(f"Canonical file changed outside this pending batch: {path}")
        write_json({"batch_id": batch_id, "input": str(input_path), "identity": identity}, pending)
        try:
            for key, path in paths.items():
                source = job / manifest["snapshots"][key]
                if fingerprint(path) != fingerprint(source):
                    atomic_write(path, lambda temp, source=source: shutil.copyfile(source, temp))
            manifest.update(state="local_committed", last_attempt=now())
            write_json(manifest, manifest_path)
            delta = read_jsonl(job / "delta_documents.jsonl")
            if delta:
                if updater is None:
                    from runtime.knowledge_base.index_qdrant_delta import update_qdrant
                    updater = update_qdrant
                metrics = updater(delta, collection=collection, url=url,
                                  batch_size=batch_size, max_length=max_length)
            else:
                metrics = {"documents": 0, "products": 0, "deleted": 0}
            result = {**report, "new_reviews": manifest["new_reviews"], "metrics": metrics,
                      "completed_at": now(), "artifact_dir": str(job)}
            write_json(result, job / "done.json")
            manifest.update(state="done", completed_at=result["completed_at"])
            write_json(manifest, manifest_path)
            pending.unlink(missing_ok=True)
            return result
        except Exception as exc:
            manifest.update(state="pending_retry", error_type=type(exc).__name__, last_attempt=now())
            write_json(manifest, manifest_path)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", choices=["logreg", "phobert", "phobert_onnx"], default="logreg")
    parser.add_argument("--collection", help="Existing collection (or set QDRANT_COLLECTION)")
    parser.add_argument("--url", help="Qdrant URL (or set QDRANT_URL)")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSV and report counts without writes/model/API calls")
    parser.add_argument("--revision", default="1", help="Change when model auxiliary files or pipeline rules change")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    try:
        result = run(args.input, model=args.model, collection=args.collection, url=args.url,
                     dry_run=args.dry_run, revision=args.revision,
                     batch_size=args.batch_size, max_length=args.max_length)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"KB update failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
