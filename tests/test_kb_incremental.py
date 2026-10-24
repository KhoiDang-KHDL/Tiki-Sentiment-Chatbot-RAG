"""Offline tests: temporary CSVs, fake predictors and in-memory Qdrant only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from runtime.absa.label_reviews import label_frame, prepare_reviews
from runtime.knowledge_base.aggregate_absa import aggregate
from runtime.knowledge_base.build_delta_documents import (
    build_delta_documents, replace_product_documents,
)
from runtime.knowledge_base.incremental_io import (
    ASPECTS, batch_lock, fingerprint, write_csv, write_jsonl, read_jsonl,
)
from runtime.knowledge_base.index_qdrant_delta import update_qdrant, stable_id, product_points
from runtime.knowledge_base.merge_reviews import merge_reviews
from scripts.update_kb import run


def labels(sentiment="positive"):
    return json.dumps([{"aspect": asp, "sentiment": sentiment} for asp in ASPECTS])


def review(pid=1, content="Sản phẩm tốt", sentiment="positive"):
    return {"product_id": pid, "content": content, "rating": 5,
            "clean_content": content.lower(), "label": labels(sentiment)}


def details():
    return pd.DataFrame([{"product_id": pid, "name": f"Sản phẩm {pid}",
                          "rating_average": 4.5, "review_count": 10, "price": 100000}
                         for pid in (1, 2)])


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return {"dense_vecs": [[1.0] + [0.0] * 1023 for _ in texts],
                "lexical_weights": [{"1": 0.5} for _ in texts]}


class DataTests(unittest.TestCase):
    def test_label_cleaning_empty_and_prediction_alignment(self):
        predictor = Mock()
        predictor.predict.return_value = [{asp: "positive" for asp in ASPECTS}] * 2
        frame = pd.DataFrame([review(1), review(1), review(2), review(2, "   ")])
        result = label_frame(frame, predictor=predictor)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(json.loads(result.iloc[0]["label"])), 6)
        self.assertEqual(len(predictor.predict.call_args.args[0]), 2)
        predictor.reset_mock()
        empty = label_frame(frame.iloc[0:0], predictor=predictor)
        self.assertTrue(empty.empty)
        predictor.predict.assert_not_called()

    def test_validation(self):
        for field, value in [("product_id", "bad"), ("product_id", "0"), ("rating", "x"), ("rating", 6)]:
            row = review()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                prepare_reviews(pd.DataFrame([row]))
        predictor = Mock()
        predictor.predict.return_value = []
        with self.assertRaises(ValueError):
            label_frame(pd.DataFrame([review()]), predictor=predictor)

    def test_merge_retry_and_same_text_across_products(self):
        main = pd.DataFrame([review()])
        batch = pd.DataFrame([review(), review(1, "Giao hàng chậm", "negative"), review(2)])
        merged, added, affected = merge_reviews(main, batch)
        self.assertEqual((len(merged), len(added), affected), (3, 2, [1, 2]))
        retried, added, affected = merge_reviews(merged, batch)
        self.assertEqual((len(retried), len(added), affected), (3, 0, [1, 2]))
        invalid = batch.copy()
        invalid.loc[0, "label"] = "[]"
        with self.assertRaises(ValueError):
            merge_reviews(main, invalid)

    def test_aggregate_and_build_direct_review_survives_next_batch(self):
        old = pd.DataFrame([review()])
        first = pd.DataFrame([review(1, "Giao hàng chậm", "negative")])
        merged, _, affected = merge_reviews(old, first)
        scores = aggregate(merged)
        self.assertEqual(scores.iloc[0]["quality_score"], 0.0)
        docs = build_delta_documents(details(), merged, scores, affected, first)
        direct_ids = {d["id"] for d in docs if d["metadata"].get("incremental_review")}
        self.assertEqual(len(direct_ids), 1)
        second = pd.DataFrame([review(1, "Đóng gói tốt")])
        merged, _, affected = merge_reviews(merged, second)
        next_docs = build_delta_documents(details(), merged, aggregate(merged), affected, second, docs)
        next_direct = [d for d in next_docs if d["metadata"].get("incremental_review")]
        self.assertEqual(len(next_direct), 2)
        self.assertTrue(direct_ids <= {d["id"] for d in next_direct})
        self.assertTrue(all(d["metadata"]["absa_quality_score"] == 0.333 for d in next_direct))
        with self.assertRaises(ValueError):
            build_delta_documents(details(), merged, aggregate(merged), [99])


class QdrantTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.collection = "offline_test"
        self.client.create_collection(self.collection,
            vectors_config={"dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()})
        reviews = pd.DataFrame([review(1), review(2)])
        self.docs = build_delta_documents(details(), reviews, aggregate(reviews), [1, 2])
        self.update(self.docs)

    def tearDown(self):
        self.client.close()

    def update(self, docs, model=None):
        return update_qdrant(docs, collection=self.collection, client=self.client,
                             model=model or FakeEncoder(), batch_size=2)

    def test_replaces_stale_payload_and_keeps_other_product_vectors(self):
        before = self.client.retrieve(self.collection,
            ids=[stable_id(d["id"]) for d in self.docs if d["metadata"]["product_id"] == 2],
            with_vectors=True)
        changed = [dict(d, content=d["content"] + " UPDATED") for d in self.docs
                   if d["metadata"]["product_id"] == 1 and d["doc_type"] != "review"]
        metrics = self.update(changed)
        self.assertEqual(metrics["deleted"], 1)
        self.assertEqual(set(product_points(self.client, self.collection, 1)),
                         {stable_id(d["id"]) for d in changed})
        after = self.client.retrieve(self.collection, ids=[p.id for p in before], with_vectors=True)
        self.assertEqual([p.model_dump() for p in before], [p.model_dump() for p in after])
        self.assertEqual(self.update(changed)["deleted"], 0)

    def test_upsert_failure_does_not_delete_old_points(self):
        before = product_points(self.client, self.collection, 1)
        encoder = Mock()
        encoder.encode.side_effect = RuntimeError("simulated embedding failure")
        with self.assertRaises(RuntimeError):
            self.update([d for d in self.docs if d["metadata"]["product_id"] == 1], encoder)
        self.assertEqual(before, product_points(self.client, self.collection, 1))

    def test_partial_upsert_can_be_retried(self):
        affected = [dict(d, content=d["content"] + " NEW") for d in self.docs
                    if d["metadata"]["product_id"] == 1]
        encoder = Mock()
        encoder.encode.side_effect = [FakeEncoder().encode(["a", "b"]), RuntimeError("second batch")]
        with self.assertRaises(RuntimeError):
            self.update(affected, encoder)
        self.assertEqual(len(product_points(self.client, self.collection, 1)), 3)
        self.update(affected)
        self.assertTrue(all(p["content"].endswith(" NEW")
                            for p in product_points(self.client, self.collection, 1).values()))

    def test_wrong_collection_rejected_before_encoder(self):
        encoder = Mock()
        with self.assertRaises(ValueError):
            update_qdrant(self.docs, collection="missing", client=self.client, model=encoder)
        encoder.encode.assert_not_called()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.processed = self.root / "data" / "processed"
        self.processed.mkdir(parents=True)
        raw = self.root / "data" / "raw"
        raw.mkdir()
        write_csv(details(), raw / "products_detail.csv")
        reviews = pd.DataFrame([review(1), review(2)])
        write_csv(reviews, self.processed / "labeled_processed_products_reviews.csv")
        scores = aggregate(reviews)
        scores.to_parquet(self.processed / "product_aspect_scores.parquet", index=False)
        docs = build_delta_documents(details(), reviews, scores, [1, 2])
        write_jsonl(docs, self.processed / "documents.jsonl")
        self.input = raw / "batch.csv"
        write_csv(pd.DataFrame([review(1, "Giao hàng chậm", "negative")]), self.input)
        self.labeler = Mock(side_effect=lambda frame, **kwargs: frame.assign(label=labels("negative")))
        self.updater = Mock(return_value={"documents": 4, "products": 1, "deleted": 0})

    def tearDown(self):
        self.directory.cleanup()

    def run_batch(self, **kwargs):
        return run(self.input, root=self.root, collection="offline", url="http://offline",
                   labeler=self.labeler, updater=self.updater, **kwargs)

    def test_dry_run_writes_nothing_and_does_not_load_models(self):
        before = {str(p): fingerprint(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.run_batch(dry_run=True)
        after = {str(p): fingerprint(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["new_reviews"], 1)
        self.labeler.assert_not_called()
        self.updater.assert_not_called()

    def test_preflight_client_without_context_manager_is_closed(self):
        # QdrantClient versions without __enter__/__exit__ must work too.
        client = Mock(spec=QdrantClient)
        with patch("qdrant_client.QdrantClient", return_value=client), \
                patch("runtime.knowledge_base.index_qdrant_delta.check_collection") as check, \
                patch("runtime.knowledge_base.index_qdrant_delta.update_qdrant", self.updater):
            result = run(self.input, root=self.root, collection="offline", url="http://offline",
                         labeler=self.labeler)
        self.assertEqual(result["new_reviews"], 1)
        check.assert_called_once_with(client, "offline")
        client.close.assert_called_once()
        self.updater.assert_called_once()

    def test_failed_preflight_closes_client_before_any_data_changes(self):
        client = Mock(spec=QdrantClient)
        before = fingerprint(self.processed / "labeled_processed_products_reviews.csv")
        with patch("qdrant_client.QdrantClient", return_value=client), \
                patch("runtime.knowledge_base.index_qdrant_delta.check_collection",
                      side_effect=ValueError("wrong collection")):
            with self.assertRaisesRegex(ValueError, "wrong collection"):
                run(self.input, root=self.root, collection="offline", url="http://offline",
                    labeler=self.labeler)
        client.close.assert_called_once()
        self.labeler.assert_not_called()
        self.assertEqual(before, fingerprint(self.processed / "labeled_processed_products_reviews.csv"))
        self.assertFalse((self.processed / "kb_batches" / "pending.json").exists())

    def test_success_backups_export_and_completed_batch_skip(self):
        result = self.run_batch()
        self.assertEqual(result["new_reviews"], 1)
        job = Path(result["artifact_dir"])
        self.assertTrue((job / "done.json").exists())
        self.assertTrue((job / "before_labeled_processed_products_reviews.csv").exists())
        docs = read_jsonl(self.processed / "documents.jsonl")
        self.assertTrue(any(d["metadata"].get("incremental_review") for d in docs))
        self.assertTrue(self.run_batch()["already_done"])
        self.assertEqual(self.updater.call_count, 1)

    def test_index_failure_retry_and_other_batch_blocked(self):
        self.updater.side_effect = RuntimeError("simulated Qdrant outage")
        with self.assertRaises(RuntimeError):
            self.run_batch()
        self.assertTrue((self.processed / "kb_batches" / "pending.json").exists())
        self.assertFalse(list((self.processed / "kb_batches").glob("*/done.json")))
        second_input = self.input.with_name("another.csv")
        write_csv(pd.DataFrame([review(2, "Giá hợp lý")]), second_input)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            run(second_input, root=self.root, collection="offline", url="http://offline",
                labeler=self.labeler, updater=self.updater)
        self.updater.side_effect = None
        result = self.run_batch()
        self.assertEqual(self.labeler.call_count, 1)
        self.assertEqual(len(pd.read_csv(self.processed / "labeled_processed_products_reviews.csv")), 3)
        self.assertFalse((self.processed / "kb_batches" / "pending.json").exists())
        self.assertEqual(result["new_reviews"], 1)

    def test_external_edit_and_corrupt_artifacts_block_retry(self):
        self.updater.side_effect = RuntimeError("outage")
        with self.assertRaises(RuntimeError):
            self.run_batch()
        snapshot = next((self.processed / "kb_batches").glob("*/delta_documents.jsonl"))
        snapshot.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "artifact"):
            self.run_batch()

    def test_missing_detail_rejected_without_canonical_writes(self):
        write_csv(pd.DataFrame([review(99)]), self.input)
        before = fingerprint(self.processed / "labeled_processed_products_reviews.csv")
        with self.assertRaisesRegex(ValueError, "Missing product details"):
            self.run_batch()
        self.assertEqual(before, fingerprint(self.processed / "labeled_processed_products_reviews.csv"))
        self.updater.assert_not_called()

    def test_empty_clean_batch_no_index(self):
        write_csv(pd.DataFrame([review(1, "  ")]), self.input)
        result = self.run_batch()
        self.assertEqual(result["new_reviews"], 0)
        self.updater.assert_not_called()

    def test_lock_released_and_second_writer_rejected(self):
        path = self.root / "lock"
        with batch_lock(path):
            with self.assertRaises(RuntimeError):
                with batch_lock(path):
                    self.fail("Acquired an already-held lock")
        with batch_lock(path):
            pass


if __name__ == "__main__":
    unittest.main()
