import json
import logging
import os
import datetime
import pandas as pd

import boto3
from botocore.exceptions import ClientError

from pipeline import (
    OUTPUT_BUCKET,
    INPUT_BUCKET,
    CHUNK_SIZE,
    _get_done_chunks,
    _chunkdone_s3_key,
    download_from_s3,
    upload_to_s3,
)

log = logging.getLogger()
log.setLevel(logging.INFO)

s3_client = boto3.client("s3")


def _list_all_chunkmeta() -> list[dict]:
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        objects   = []
        for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix="checkpoints/chunks/meta_"):
            objects.extend(page.get("Contents", []))
        return objects
    except ClientError:
        return []


def _upload_next_chunk(meta: dict, meta_created_time: datetime.datetime, now: datetime.datetime) -> str | None:
    base_name    = meta["base_name"]
    total_chunks = meta["total_chunks"]

    done_chunks  = _get_done_chunks(base_name, total_chunks)
    next_idx     = len(done_chunks)

    if next_idx >= total_chunks:
        log.info("%s: tất cả %d chunks đã done. Bỏ qua.", base_name, total_chunks)
        return None

    today_utc_str = now.strftime("%Y-%m-%d")

    if next_idx > 0:
        last_done_idx = next_idx - 1
        last_done_key = _chunkdone_s3_key(base_name, last_done_idx)
        try:
            done_obj = s3_client.get_object(Bucket=OUTPUT_BUCKET, Key=last_done_key)
            done_data = json.loads(done_obj["Body"].read().decode("utf-8"))

            if done_data.get("done_date") == today_utc_str:
                log.info("%s: Chunk %03d đã chạy xong hôm nay (%s). Bỏ qua chiều nay.",
                         base_name, last_done_idx, today_utc_str)
                return None
        except ClientError:
            log.warning("%s: Không đọc được JSON file done %03d. Bỏ qua để an toàn.", base_name, last_done_idx)
            return None
    else:
        meta_created_date_str = meta_created_time.strftime("%Y-%m-%d")
        if meta_created_date_str == today_utc_str:
            log.info("%s: File gốc mới upload và đang chạy chunk_000 hôm nay (%s). Bỏ qua chiều nay.",
                     base_name, today_utc_str)
            return None

    next_key   = meta["chunk_keys"][next_idx]
    next_local = f"/tmp/{os.path.basename(next_key)}"

    try:
        s3_client.head_object(Bucket=INPUT_BUCKET, Key=next_key)
        log.warning("Chunk %d đã có trên S3 nhưng chưa DONE. Upload đè để re-trigger.", next_idx)
    except ClientError:
        pass

    orig_filename = os.path.basename(meta["original_key"])
    orig_local    = f"/tmp/{orig_filename}"
    download_from_s3(INPUT_BUCKET, meta["original_key"], orig_local)

    df_orig   = pd.read_csv(orig_local)
    start_row = next_idx * CHUNK_SIZE
    df_chunk  = df_orig.iloc[start_row: start_row + CHUNK_SIZE]
    df_chunk.to_csv(next_local, index=False)

    upload_to_s3(next_local, INPUT_BUCKET, next_key)
    log.info("Scheduler uploaded chunk_%03d -> s3://%s/%s", next_idx, INPUT_BUCKET, next_key)
    return next_key


def handler(event, context):
    log.info("Chunk scheduler started.")
    chunkmeta_objects = _list_all_chunkmeta()

    if not chunkmeta_objects:
        return {"statusCode": 200, "body": "no chunks pending"}

    chunkmeta_objects.sort(key=lambda x: x["LastModified"])

    now = datetime.datetime.now(datetime.timezone.utc)
    results = []

    for obj in chunkmeta_objects:
        try:
            raw  = s3_client.get_object(Bucket=OUTPUT_BUCKET, Key=obj["Key"])
            meta = json.loads(raw["Body"].read())

            uploaded = _upload_next_chunk(meta, obj["LastModified"], now)

            if uploaded:
                results.append({"base_name": meta["base_name"], "uploaded": uploaded})
        except Exception as e:
            log.exception("Lỗi khi xử lý %s: %s", obj["Key"], e)

    return {"statusCode": 200, "body": json.dumps(results)}
