"""
DAG 3: ABSA Inference Pipeline
Input:  data/processed/reviews_clean_YYYYMMDD.csv
Output: data/processed/labeled_reviews_YYYYMMDD.csv
Model:  PhoBERT v2 (accuracy first cho KB rebuild)
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_DIR)

default_args = {
    'owner': 'H26NG',
    'start_date': datetime(2026, 5, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_clean_reviews(**context):
    import glob
    from pathlib import Path
    import pandas as pd

    files = sorted(glob.glob("/opt/airflow/data/processed/reviews_clean_*.csv"))
    if not files:
        print("Không tìm thấy file clean nào — skip")
        return False

    # Tìm tất cả file chưa inference
    pending = []
    for f in files:
        today        = Path(f).stem.replace("reviews_clean_", "")
        labeled_path = Path(f"/opt/airflow/data/processed/labeled_reviews_{today}.csv")
        if not labeled_path.exists():
            pending.append({"clean_path": f, "today": today})

    if not pending:
        print("Tất cả files đã được inference — skip")
        return False

    context['ti'].xcom_push(key='pending_clean', value=pending)
    context['ti'].xcom_push(key='clean_path',    value=pending[-1]["clean_path"])
    context['ti'].xcom_push(key='today',         value=pending[-1]["today"])
    print(f"Tìm thấy {len(pending)} files chưa inference → tiếp tục")
    return True

def run_absa_inference(**context):
    from pathlib import Path
    import pandas as pd
    import json
    from src.absa.inference import get_phobert

    ti            = context['ti']
    pending_clean = ti.xcom_pull(key='pending_clean', task_ids='check_clean_reviews') or []

    ASPECTS   = ["description", "quality", "packaging", "delivery", "service", "price"]
    predictor = get_phobert(version='v2')

    def make_label(pred):
        return json.dumps(
            [{"aspect": asp, "sentiment": pred[asp]} for asp in ASPECTS],
            ensure_ascii=False
        )

    for p in pending_clean:
        df    = pd.read_csv(p["clean_path"])
        texts = df['clean_content'].fillna('').tolist()

        print(f"PhoBERT inference trên {len(texts)} reviews ({p['today']})...")
        predictions = predictor.predict(texts)
        df['label'] = [make_label(pred) for pred in predictions]

        output_path = Path(f"/opt/airflow/data/processed/labeled_reviews_{p['today']}.csv")
        df.to_csv(output_path, index=False)
        print(f"{p['today']}: {len(df)} reviews labeled")

def check_retrain_trigger(**context):
    """Check 2 metrics và notify team ABSA nếu cần retrain."""
    import glob
    import pandas as pd
    import mlflow
    import wandb
    from pathlib import Path

    trigger = False
    reasons = []

    # ── Metric 1: Đủ data mới chưa? (>= 100 reviews) ──
    total_new = 0
    for f in glob.glob("/opt/airflow/data/processed/labeled_reviews_*.csv"):
        try:
            total_new += len(pd.read_csv(f))
        except Exception:
            pass

    if total_new >= 250:
        trigger = True
        reasons.append(f"Data volume: {total_new} reviews mới >= 250")

    print(f"Data check: {total_new} reviews mới {'TRIGGER' if total_new >= 250 else 'OK'}")

    # ── Metric 2: F1 có drop không? ──
    baseline_f1 = 0.848  # PhoBERT v2 baseline
    try:
        api  = wandb.Api()
        runs = api.runs(
            f"{os.getenv('WANDB_ENTITY', 'cs317-mlops-org')}/sentiment-chatbot-mlops",
            filters={"group": "phobert"}
        )
        if runs:
            latest_f1 = max(r.summary.get("avg_f1", 0) for r in runs)
            if latest_f1 < baseline_f1 * 0.95:
                trigger = True
                reasons.append(f"Performance drift: F1 {latest_f1:.3f} < threshold {baseline_f1 * 0.95:.3f}")
            print(f"F1 check: {latest_f1:.3f} {'TRIGGER' if latest_f1 < baseline_f1 * 0.95 else 'OK'}")
    except Exception as e:
        print(f"W&B F1 check failed: {e} → skip")

    # ── Log vào MLflow ──
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("retrain_monitoring")
    with mlflow.start_run(run_name=f"retrain_check_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.log_metrics({
            "new_reviews_count": total_new,
            "retrain_triggered": 1 if trigger else 0,
        })

    # ── Notify nếu cần retrain ──
    if trigger:
        reason_text = "\n".join(reasons)
        print(f"RETRAIN TRIGGERED:\n{reason_text}")

        # W&B Alert
        try:
            wandb.init(
                project="sentiment-chatbot-mlops",
                entity=os.getenv("WANDB_ENTITY", "cs317-mlops-org"),
                name=f"retrain_alert_{datetime.now().strftime('%Y%m%d')}",
            )
            wandb.log({"retrain_alert": reason_text})
            wandb.finish()
            print("W&B alert logged!")
        except Exception as e:
            print(f"W&B alert failed: {e}")
    else:
        print(f"Chưa cần retrain (data={total_new}, baseline_f1={baseline_f1})")

def pull_latest_absa_model(**context):
    """Pull PhoBERT model mới nhất từ W&B về local."""
    import wandb
    from pathlib import Path

    model_dir = Path("/opt/airflow/models/absa/v2/phobert")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check nếu model đã có rồi thì skip
    if (model_dir / "phobert.pt").exists():
        print("Model đã có local → skip download")
        return

    api      = wandb.Api()
    artifact = api.artifact("cs317-mlops-org/sentiment-chatbot-mlops/phobert-v2:latest")
    artifact.download(root=str(model_dir))
    print(f"Downloaded PhoBERT model → {model_dir}")

with DAG(
    'absa_inference_pipeline',
    default_args=default_args,
    description='PhoBERT ABSA inference trên reviews mới',
    schedule='30 19 * * *',
    catchup=False,
) as dag:

    t_check = ShortCircuitOperator(
        task_id='check_clean_reviews',
        python_callable=check_clean_reviews,
    )

    t_pull = PythonOperator(
        task_id='pull_latest_absa_model',
        python_callable=pull_latest_absa_model,
    )

    t_infer = PythonOperator(
        task_id='run_absa_inference',
        python_callable=run_absa_inference,
    )

    t_retrain = PythonOperator(
        task_id='check_retrain_trigger',
        python_callable=check_retrain_trigger,
    )

    t_check >> t_pull >> t_infer >> t_retrain
