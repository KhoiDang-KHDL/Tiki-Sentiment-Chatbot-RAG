"""
DAG 4: KB Incremental Update Pipeline

Logic:
- Có labeled_reviews_YYYYMMDD.csv pending thì lấy product_id trong file đó làm affected products.
- Merge review mới vào main CSV nếu chưa tồn tại.
- Dù review đã merge trước đó nhưng KB chưa update, vẫn update KB theo product_id pending.
- Aggregate ABSA lại để score mới đúng.
- Build delta documents cho affected products.
- Chỉ update Qdrant cho affected products.
- Sau khi KB update thành công thì archive labeled_reviews_*.csv.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


default_args = {
    "owner": "chatbot_team",
    "start_date": datetime(2026, 5, 24),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_and_merge_labeled(**context):
    import glob
    from pathlib import Path
    import pandas as pd

    processed_dir = Path("/opt/airflow/data/processed")
    archive_dir = processed_dir / "kb_archive"

    labeled_files = sorted(glob.glob(str(processed_dir / "labeled_reviews_*.csv")))
    main_path = processed_dir / "labeled_processed_products_reviews.csv"

    delta_reviews_path = processed_dir / "kb_delta_reviews.csv"
    affected_ids_path = processed_dir / "kb_delta_product_ids.txt"

    # Clean old delta artifacts
    if delta_reviews_path.exists():
        delta_reviews_path.unlink()
    if affected_ids_path.exists():
        affected_ids_path.unlink()

    if not labeled_files:
        print("Không có labeled_reviews_*.csv pending → skip")
        context["ti"].xcom_push(key="new_count", value=0)
        context["ti"].xcom_push(key="total_reviews", value=0)
        context["ti"].xcom_push(key="affected_products", value=0)
        context["ti"].xcom_push(key="processed_labeled_files", value=[])
        return True

    if not main_path.exists():
        raise FileNotFoundError(f"Không tìm thấy main CSV: {main_path}")

    required_cols = {"product_id", "content"}

    df_main = pd.read_csv(main_path)
    before = len(df_main)

    if not required_cols.issubset(df_main.columns):
        raise ValueError(f"Main file thiếu cột bắt buộc: {required_cols - set(df_main.columns)}")

    df_main["product_id"] = df_main["product_id"].astype(str)
    df_main["content"] = df_main["content"].astype(str)

    existing = set(zip(df_main["product_id"], df_main["content"]))

    new_parts = []
    pending_parts = []
    affected_ids_from_files = set()
    valid_labeled_files = []

    for f in labeled_files:
        path = Path(f)
        df = pd.read_csv(path)

        if not required_cols.issubset(df.columns):
            print(f"Skip {path.name}: thiếu cột {required_cols - set(df.columns)}")
            continue

        df["product_id"] = df["product_id"].astype(str)
        df["content"] = df["content"].astype(str)

        # Quan trọng:
        # affected_products lấy từ file pending, không phụ thuộc review có mới hoàn toàn hay không.
        # Vì có case review đã merge vào main nhưng KB update fail/skip ở lần trước.
        file_product_ids = df["product_id"].dropna().astype(int).unique().tolist()
        affected_ids_from_files.update(file_product_ids)

        df["_source_file"] = path.name
        pending_parts.append(df.copy())
        valid_labeled_files.append(str(path))

        mask_new = [
            (pid, content) not in existing
            for pid, content in zip(df["product_id"], df["content"])
        ]

        new = df.loc[mask_new].copy()

        if len(new) > 0:
            new_parts.append(new)

            df_main = pd.concat(
                [df_main, new.drop(columns=["_source_file"], errors="ignore")],
                ignore_index=True,
            )

            existing.update(zip(new["product_id"], new["content"]))
            print(f"Merged {path.stem}: +{len(new)} reviews")
        else:
            print(f"{path.stem}: không có review mới trong main CSV, nhưng vẫn mark product_id để update KB")

    affected_ids = sorted(int(x) for x in affected_ids_from_files)
    added = len(df_main) - before

    if valid_labeled_files:
        context["ti"].xcom_push(key="processed_labeled_files", value=valid_labeled_files)
    else:
        context["ti"].xcom_push(key="processed_labeled_files", value=[])

    if not affected_ids:
        print("Không có product_id nào trong labeled files → skip downstream")
        context["ti"].xcom_push(key="new_count", value=0)
        context["ti"].xcom_push(key="total_reviews", value=len(df_main))
        context["ti"].xcom_push(key="affected_products", value=0)
        return True

    # Save main only if actually changed
    if added > 0:
        df_main.to_csv(main_path, index=False)
        print(f"Tổng reviews main: {before} → {len(df_main)}")
    else:
        print(f"Main reviews không đổi: {len(df_main)} reviews")

    # Save all pending labeled rows for traceability, not only new rows.
    # Delta build chỉ cần affected product ids, nhưng file này giúp debug/log.
    if pending_parts:
        delta_df = pd.concat(pending_parts, ignore_index=True)
        delta_df = delta_df.drop_duplicates(subset=["product_id", "content"])
        delta_df.to_csv(delta_reviews_path, index=False)
        print(f"Saved pending delta reviews → {delta_reviews_path} ({len(delta_df)} rows)")

    affected_ids_path.write_text(
        "\n".join(str(x) for x in affected_ids),
        encoding="utf-8",
    )

    print(f"New reviews merged: {added}")
    print(f"Affected products for KB update: {len(affected_ids)}")
    print(f"Saved affected ids → {affected_ids_path}")

    context["ti"].xcom_push(key="new_count", value=added)
    context["ti"].xcom_push(key="total_reviews", value=len(df_main))
    context["ti"].xcom_push(key="affected_products", value=len(affected_ids))

    return True


def has_affected_products(**context):
    ti = context["ti"]

    affected_products = ti.xcom_pull(
        key="affected_products",
        task_ids="check_labeled_reviews",
    ) or 0

    new_count = ti.xcom_pull(
        key="new_count",
        task_ids="check_labeled_reviews",
    ) or 0

    print(f"new_count={new_count}")
    print(f"affected_products={affected_products}")

    if affected_products <= 0:
        print("Không có product nào cần update KB → skip aggregate/build/index")
        return False

    print("Có product_id cần update KB → tiếp tục incremental KB update")
    return True


def run_aggregate_absa(**context):
    from src.kb.aggregate_absa import main

    main()
    print("Stage 1: aggregate_absa done")


def run_build_delta_documents(**context):
    from src.kb.build_delta_documents import main

    main()
    print("Stage 2-delta: build_delta_documents done")


def run_smoke_test(**context):
    from src.chatbot.retrieval import TikiRAG

    rag = TikiRAG()

    queries = [
        "cốc giữ nhiệt chất lượng tốt",
        "đồ gia dụng giá rẻ",
        "sản phẩm giao hàng nhanh",
        "pin tiểu chính hãng",
        "bình nước inox bền",
    ]

    passed = 0

    for q in queries:
        docs = rag.search(q, top_k=3)

        if len(docs) >= 3:
            passed += 1
            print(f"PASS | '{q}' → {len(docs)} docs")
        else:
            print(f"FAIL | '{q}' → chỉ {len(docs)} docs")

    print(f"Smoke test: {passed}/{len(queries)} passed")
    context["ti"].xcom_push(key="smoke_passed", value=passed)

    return passed


def log_to_mlflow(**context):
    import json
    from pathlib import Path
    import mlflow

    ti = context["ti"]

    new_count = ti.xcom_pull(
        key="new_count",
        task_ids="check_labeled_reviews",
    ) or 0

    total_reviews = ti.xcom_pull(
        key="total_reviews",
        task_ids="check_labeled_reviews",
    ) or 0

    affected_products = ti.xcom_pull(
        key="affected_products",
        task_ids="check_labeled_reviews",
    ) or 0

    smoke_passed = ti.xcom_pull(
        key="smoke_passed",
        task_ids="run_smoke_test",
    ) or 0

    metrics = {
        "new_reviews": new_count,
        "total_reviews": total_reviews,
        "affected_products": affected_products,
        "smoke_passed": smoke_passed,
    }

    # Optional: nếu index_qdrant_delta.py có ghi metrics file thì log thêm.
    metrics_path = Path("/opt/airflow/data/processed/kb_delta_metrics.json")
    if metrics_path.exists():
        try:
            extra = json.loads(metrics_path.read_text(encoding="utf-8"))
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
            print(f"Loaded extra KB metrics from {metrics_path}")
        except Exception as e:
            print(f"Cannot read extra metrics file: {e}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("kb_incremental_update_pipeline")

    with mlflow.start_run(run_name=f"kb_incremental_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("absa_model", "phobert_v2")
        mlflow.log_param("update_mode", "incremental")
        mlflow.log_param("update_date", datetime.now().strftime("%Y-%m-%d"))

    print("MLflow logged")
    print(f"Logged metrics: {metrics}")


def archive_labeled_files(**context):
    import shutil
    from pathlib import Path

    ti = context["ti"]

    files = ti.xcom_pull(
        key="processed_labeled_files",
        task_ids="check_labeled_reviews",
    ) or []

    if not files:
        print("Không có labeled file nào để archive")
        return

    archive_dir = Path("/opt/airflow/data/processed/kb_archive")
    archive_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        src = Path(f)

        if not src.exists():
            print(f"Skip archive, file không còn tồn tại: {src}")
            continue

        dst = archive_dir / src.name

        # Nếu đã có file cùng tên trong archive, thêm timestamp để không overwrite.
        if dst.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = archive_dir / f"{src.stem}_{ts}{src.suffix}"

        shutil.move(str(src), str(dst))
        print(f"Archived {src.name} → {dst}")


with DAG(
    dag_id="kb_rebuild_pipeline",
    default_args=default_args,
    description="Incrementally update Qdrant KB từ labeled reviews pending",
    schedule="0 20 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["kb", "qdrant", "rag", "incremental"],
) as dag:

    t_check = PythonOperator(
        task_id="check_labeled_reviews",
        python_callable=check_and_merge_labeled,
        execution_timeout=timedelta(minutes=10),
    )

    t_has_affected = ShortCircuitOperator(
        task_id="has_affected_products",
        python_callable=has_affected_products,
    )

    t_stage1 = PythonOperator(
        task_id="run_aggregate_absa",
        python_callable=run_aggregate_absa,
        execution_timeout=timedelta(minutes=30),
    )

    t_stage2 = PythonOperator(
        task_id="run_build_delta_documents",
        python_callable=run_build_delta_documents,
        execution_timeout=timedelta(minutes=15),
    )

    t_stage3 = BashOperator(
        task_id="run_index_qdrant_delta",
        bash_command="""
        set -e
        cd /opt/airflow
        export PYTHONPATH=/opt/airflow
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export EMBEDDING_DEVICE=cpu
        export KB_INDEX_BATCH_SIZE=4
        export KB_MAX_LENGTH=1024

        echo "[Stage 3-delta] Start incremental Qdrant update..."
        python -u /opt/airflow/src/kb/index_qdrant_delta.py
        echo "[Stage 3-delta] incremental index done"
        """,
        execution_timeout=timedelta(hours=1),
    )

    t_smoke = PythonOperator(
        task_id="run_smoke_test",
        python_callable=run_smoke_test,
        execution_timeout=timedelta(minutes=15),
    )

    t_mlflow = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,
        execution_timeout=timedelta(minutes=10),
    )

    t_archive = PythonOperator(
        task_id="archive_labeled_files",
        python_callable=archive_labeled_files,
        execution_timeout=timedelta(minutes=5),
    )

    (
        t_check>> t_has_affected>> t_stage1>> t_stage2>> t_stage3>> t_smoke>> t_mlflow>> t_archive
    )
