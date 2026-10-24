from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Trỏ về gốc project để import script upload
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_DIR)

from src.crawling.upload_s3 import sync_local_to_s3

default_args = {
    'owner': 'phi_quyen',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 23),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'tiki_only_sync_s3',
    default_args=default_args,
    description='Chỉ quét folder data/raw và đồng bộ lên S3 mỗi 1 ngày NẾU có data mới',
    schedule_interval='0 2 * * *',
    catchup=False,
) as dag:

    sync_task = PythonOperator(
        task_id='sync_raw_to_s3_only',
        python_callable=sync_local_to_s3,
        op_kwargs={
            'local_dir': os.path.join(BASE_DIR, 'data/raw'),
            's3_prefix': 'raw/'
        }
    )

    sync_task
