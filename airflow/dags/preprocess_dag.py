"""
DAG 2: Preprocess Pipeline
- Download reviews mới từ S3
- Clean + tokenize
Input:  s3://tiki-crawl-data/raw/reviews_YYYYMMDD.csv
Output: data/processed/reviews_clean_YYYYMMDD.csv
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_DIR)

default_args = {
    'owner': 'chatbot_team',
    'start_date': datetime(2026, 5, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def download_from_s3(**context):
    import boto3
    import glob
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()

    s3  = boto3.client("s3", region_name="ap-southeast-2")
    res = s3.list_objects_v2(Bucket="tiki-crawl-data", Prefix="raw/reviews_")

    if not res.get("Contents"):
        print("Không có file reviews nào trên S3 — skip")
        return False

    pending = []
    for obj in sorted(res["Contents"], key=lambda x: x["Key"]):
        s3_key   = obj["Key"]
        filename = s3_key.split("/")[-1]
        today    = filename.replace("reviews_", "").replace(".csv", "")

        # Check đã preprocess chưa
        clean_path = Path("/opt/airflow/data/processed") / f"reviews_clean_{today}.csv"
        if clean_path.exists():
            print(f"{filename} đã preprocess rồi — skip")
            continue

        # Download nếu chưa có local
        local_path = Path("/opt/airflow/data/raw") / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            s3.download_file("tiki-crawl-data", s3_key, str(local_path))
            print(f"Downloaded {filename}")

        pending.append({"raw_path": str(local_path), "today": today})

    if not pending:
        print("Tất cả files đã được preprocess — skip")
        return False

    context['ti'].xcom_push(key='pending_files', value=pending)
    context['ti'].xcom_push(key='raw_path',      value=pending[-1]["raw_path"])
    context['ti'].xcom_push(key='today',         value=pending[-1]["today"])
    print(f"Sẽ preprocess {len(pending)} files")
    return True

def run_preprocess(**context):
    """Clean + tokenize từng file chưa preprocess."""
    from pathlib import Path
    import pandas as pd
    from src.preprocessing.clean_review import clean_review

    ti = context['ti']
    pending_files = ti.xcom_pull(
        key='pending_files',
        task_ids='download_from_s3'
    ) or []

    if not pending_files:
        print("Không có file pending để preprocess")
        return

    processed_outputs = []
    total_cleaned = 0

    for f in pending_files:
        raw_path = f["raw_path"]
        today = f["today"]

        output_path = Path("/opt/airflow/data/processed") / f"reviews_clean_{today}.csv"

        if output_path.exists():
            print(f"{output_path.name} đã tồn tại → skip")
            continue

        df = pd.read_csv(raw_path)
        df_cleaned = clean_review(df, column_name='content')

        # Dedup an toàn hơn
        if "product_id" in df_cleaned.columns:
            df_cleaned.drop_duplicates(subset=["product_id", "clean_content"], inplace=True)
        else:
            df_cleaned.drop_duplicates(subset=["clean_content"], inplace=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)

        processed_outputs.append(str(output_path))
        total_cleaned += len(df_cleaned)

        print(f"Preprocess: {raw_path} → {output_path} ({len(df_cleaned)} reviews)")

    context['ti'].xcom_push(key='clean_paths', value=processed_outputs)
    context['ti'].xcom_push(key='clean_count', value=total_cleaned)

    print(f"Tổng clean mới: {total_cleaned} reviews")
    print(f"Files output: {processed_outputs}")

with DAG(
    'preprocess_pipeline',
    default_args=default_args,
    description='Download S3 + clean reviews mới mỗi ngày',
    schedule='15 19 * * *',  # 2h15 sáng VN
    catchup=False,
) as dag:

    t_download = ShortCircuitOperator(
        task_id='download_from_s3',
        python_callable=download_from_s3,
    )

    t_preprocess = PythonOperator(
        task_id='run_preprocess',
        python_callable=run_preprocess,
    )

    t_download >> t_preprocess
