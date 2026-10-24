from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import boto3
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s3_sync")

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'ap-southeast-2')
    )

def sync_local_to_s3(local_dir, s3_prefix='raw/'):
    bucket_name = os.getenv('S3_BUCKET_NAME')
    s3 = get_s3_client()

    # 1. Check file trên S3
    existing_s3_files = []
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
        if 'Contents' in response:
            existing_s3_files = [obj['Key'] for obj in response['Contents']]
    except Exception as e:
        logger.error(f"Lỗi kết nối S3: {e}")
        return

    # 2. Upload file CSV mới
    for file_name in os.listdir(local_dir):
        if file_name.endswith('.csv'):
            local_path = os.path.join(local_dir, file_name)
            s3_key = f"{s3_prefix}{file_name}"

            if s3_key not in existing_s3_files:
                logger.info(f"Đang đẩy: {file_name}")
                s3.upload_file(local_path, bucket_name, s3_key)
            else:
                logger.info(f"Đã có: {file_name}, bỏ qua.")

if __name__ == "__main__":
    raw_dir = os.path.join(os.getcwd(), 'data', 'raw')

    sync_local_to_s3(raw_dir)
