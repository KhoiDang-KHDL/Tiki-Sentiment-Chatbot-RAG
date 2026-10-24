import json
import logging
import urllib.parse
import os

from botocore.exceptions import ClientError

from pipeline import (
    run_pipeline,
    OUTPUT_BUCKET,
    _chunkdone_s3_key,
    s3_client
)

log = logging.getLogger()
log.setLevel(logging.INFO)

SKIP_PREFIXES  = ["processed/", "labeled/", "checkpoints/", "final/"]
TRIGGER_PREFIX = os.environ.get("TRIGGER_PREFIX", "raw/")

def _process_s3_record(bucket: str, key: str) -> dict:
    log.info("Received S3 event: s3://%s/%s", bucket, key)

    ABSOLUTE_SKIP = ["processed/", "labeled/", "checkpoints/"]
    if any(key.startswith(pfx) for pfx in ABSOLUTE_SKIP):
        log.info("File nằm trong danh sách skip prefix. Skip.")
        return {"key": key, "status": "skipped"}

    if not key.lower().endswith(".csv") or "reviews" not in os.path.basename(key).lower():
        return {"key": key, "status": "skipped"}

    filename = os.path.basename(key)

    if "chunks/" in key or "_chunk_" in filename:
        try:
            base_name = filename.rsplit("_chunk_", 1)[0]
            chunk_idx = int(filename.rsplit("_chunk_", 1)[1].split(".")[0])
            done_key = _chunkdone_s3_key(base_name, chunk_idx)

            s3_client.head_object(Bucket=OUTPUT_BUCKET, Key=done_key)
            log.info("Chunk %d của %s ĐÃ xử lý xong trước đó rồi. Skip.", chunk_idx, base_name)
            return {"key": key, "status": "skipped"}
        except ClientError:
            log.info("Đúng luồng: Kích hoạt Pipeline xử lý mô hình cho chunk: %s", filename)
            try:
                output_key = run_pipeline(bucket, key)
                return {"key": key, "status": "success_process_chunk", "output": output_key}
            except Exception as exc:
                log.exception("Pipeline dán nhãn chunk thất bại cho %s: %s", key, exc)
                raise
        except (IndexError, ValueError):
            return {"key": key, "status": "error_bad_chunk_name"}

    else:
        if TRIGGER_PREFIX and not key.startswith(TRIGGER_PREFIX):
            return {"key": key, "status": "skipped"}

        log.info("Phát hiện file gốc mới. Kích hoạt tiến trình bẻ file (Chunking): %s", filename)
        try:
            output_key = run_pipeline(bucket, key)
            return {"key": key, "status": "success_split_file", "output": output_key}
        except Exception as exc:
            log.exception("Pipeline chia chunk file gốc thất bại cho %s: %s", key, exc)
            raise

def lambda_handler(event, context):
    results = []

    if "Records" in event and event["Records"] and event["Records"][0].get("eventSource") == "aws:sqs":
        for sqs_record in event["Records"]:
            body           = json.loads(sqs_record["body"])

            if "Event" in body and body["Event"] == "s3:TestEvent":
                log.info("Skipping S3 test event.")
                continue

            for s3_record in body.get("Records", []):
                bucket = s3_record["s3"]["bucket"]["name"]
                key    = urllib.parse.unquote_plus(s3_record["s3"]["object"]["key"])
                result = _process_s3_record(bucket, key)
                results.append(result)

    elif "Records" in event:
        for record in event["Records"]:
            if "s3" not in record:
                continue
            bucket = record["s3"]["bucket"]["name"]
            key    = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
            result = _process_s3_record(bucket, key)
            results.append(result)

    return {
        "statusCode": 200,
        "body": json.dumps(results, ensure_ascii=False),
    }
