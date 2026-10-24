from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# qa bun ngu rôi
# Tự động lấy path của project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_DIR)

from src.crawling.scheduler import crawl_changed_products

default_args = {
    'owner': 'phi_quyen',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 23),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'tiki_incremental_crawler',
    default_args=default_args,
    description='Crawl bù review mới mỗi ngày',
    schedule_interval='0 1 * * *', # chạy mỗi ngày lúc 2h sáng (19 h là h utc )
    catchup=False, # Không chạy bù cho những ngày quá khứ
) as dag:

    run_incremental = PythonOperator(
        task_id='crawl_new_reviews',
        python_callable=crawl_changed_products,
        op_kwargs={
            'products_csv': os.path.join(BASE_DIR, 'data/raw/products_list.csv')
        }
    )
