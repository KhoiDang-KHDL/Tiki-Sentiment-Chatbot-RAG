# 🛒 Tiki Sentiment Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.7-017CEE)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://www.docker.com/)
[![W&B](https://img.shields.io/badge/Tracking-W%26B-FFBE00)](https://wandb.ai/cs317-mlops-org)

Hệ thống hỏi đáp sản phẩm Tiki kết hợp **Retrieval-Augmented Generation (RAG)** và
**Aspect-Based Sentiment Analysis (ABSA)**. Repository gồm các module thu thập dữ liệu,
huấn luyện/inference, cập nhật Knowledge Base, phục vụ chatbot và kiểm tra vận hành.
Sáu khía cạnh ABSA: chất lượng, giá cả, mô tả, dịch vụ, đóng gói và giao hàng.

> Các hình minh họa và badge được giữ nguyên. Phiên bản, entry point và flow thực thi
> cụ thể được mô tả dưới đây theo code/config trong repository.

## 1. Kiến trúc hệ thống

![Kiến trúc tổng quan](docs/images/architecture_overview.jpg)

Các nhóm chức năng trong repo:

- **Data pipeline:** crawler → CSV → S3 → preprocessing → ABSA inference → KB update.
- **Serving HTTP:** Chainlit → FastAPI → retrieval/Qdrant → LLM; URL analysis có context riêng.
- **Model development:** chuẩn bị dataset, train/sweep, evaluation và export ONNX.
- **Operations:** Docker Compose, Prometheus/Grafana, Locust, MLflow/W&B và GitHub Actions CI.

![Tech Stack](docs/images/tech_stack.jpg)

| Entry point / cấu hình | Chức năng thực thi |
|---|---|
| `docker-compose.runtime.yml` | Chạy `runtime/`: HTTP API/UI, Qdrant, metrics/dashboard và load-test profile |
| `docker-compose.yml` | Chạy các service dùng `src/`, Airflow/Celery, PostgreSQL, Redis và MLflow |
| `scripts/update_kb.py` | Incremental writer sử dụng `runtime/knowledge_base/` |
| `airflow/dags/` | Năm DAG xử lý crawl, S3, clean, inference và KB |
| `.github/workflows/ci.yml` | Test/static checks và container smoke trên GitHub-hosted runner |

Hai Compose là các cấu hình độc lập; không tự include hoặc khởi động lẫn nhau.
Chọn cấu hình phù hợp và tránh chạy trùng host ports hoặc nhiều writer trên cùng KB.

| Thành phần | Phiên bản/cấu hình trong repo |
|---|---|
| API/UI runtime containers | Python 3.11; FastAPI 0.148.0; Chainlit 2.12.0 |
| `docker/api.Dockerfile` | Python 3.11.5; `requirements-api.txt` pin FastAPI 0.115.0 |
| Airflow | 2.7.3 trong `Dockerfile.airflow`, CeleryExecutor |
| CI | Python 3.11 và 3.12 |
| Retrieval | BGE-M3, dense 1.024 chiều, Cosine; ONNX local tùy chọn |

## 2. Data Processing Pipeline

![Data Pipeline](docs/images/data_pipeline.jpg)

### 2.1. Thu thập và điều phối dữ liệu

`src/crawling/main.py` điều phối các module danh sách sản phẩm, chi tiết và review.
`src/crawling/scheduler.py` kiểm tra thay đổi và ghi `data/raw/reviews_YYYYMMDD.csv`.
Các file raw được đồng bộ lên S3 trước bước preprocessing.

| DAG ID | Cron trong code | Công việc |
|---|---|---|
| `tiki_incremental_crawler` | `0 1 * * *` | Kiểm tra sản phẩm trong products_list.csv, crawl review mới và cập nhật state |
| `tiki_only_sync_s3` | `0 2 * * *` | Upload CSV trong data/raw lên S3 nếu object key chưa tồn tại; không trực tiếp crawl |
| `preprocess_pipeline` | `15 19 * * *` | Tìm raw/reviews_*.csv trên S3, download, clean/dedup và ghi reviews_clean_*.csv |
| `absa_inference_pipeline` | `30 19 * * *` | Load/download PhoBERT v2 PyTorch, inference và ghi labeled_reviews_*.csv |
| `kb_rebuild_pipeline` | `0 20 * * *` | Merge pending reviews, aggregate, build delta, index, smoke check, log và archive |

Cron được diễn giải theo timezone của Airflow; không tự suy ra giờ Việt Nam từ comment.
Các DAG có lịch riêng và kiểm tra file đầu vào, không có sensor/trigger nối chéo giữa DAG.
Thứ tự trên mô tả phụ thuộc dữ liệu, không phải một DAG chain được bảo đảm tự động.

Preprocess hiện dùng bucket `tiki-crawl-data` và region `ap-southeast-2` trong code.
Uploader đọc `S3_BUCKET_NAME` và `AWS_REGION`; cần đồng bộ các giá trị trước khi chạy.
Thông tin W&B artifact trong inference DAG cũng cần phù hợp với tài khoản được cấu hình.

### 2.2. Inference và retrain monitoring

Inference DAG gọi `get_phobert(version="v2")`, dùng checkpoint `phobert.pt` và
`clean_content`; không gọi predictor ONNX trong task này.

Task `check_retrain_trigger`:

- Đếm tổng review trong các file `labeled_reviews_*.csv`; trigger khi đạt **250**.
- Lấy giá trị `avg_f1` cao nhất từ các W&B run được truy vấn, so với `0.848 × 0.95`.
- Ghi `new_reviews_count`/`retrain_triggered` vào MLflow và log thông tin lên W&B.

Đây là kiểm tra điều kiện từ file/run history. Task chưa tính lại F1 trên batch mới
và chưa tự chạy một training job.

### 2.3. Cập nhật Knowledge Base

`kb_rebuild_pipeline` xử lý:

```text
pending labeled reviews → merge main CSV → affected product IDs
→ aggregate ABSA → build delta documents → src/kb/index_qdrant_delta.py
→ retrieval smoke test → MLflow → archive labeled files
```

Writer `scripts/update_kb.py` cung cấp luồng CSV độc lập:

```text
validate/clean → ABSA labels → deduplicate/merge → aggregate → delta documents
→ local snapshots → embed/upsert → remove stale IDs → verify → done
```

Luồng này có OS lock, backup, manifest/hash checks và retry bằng stable IDs.
Chỉ embed lại affected products. Dữ liệu canonical gồm:

- `data/processed/labeled_processed_products_reviews.csv`.
- `data/processed/product_aspect_scores.parquet`.
- `data/processed/documents.jsonl`.

Batch artifacts nằm trong `data/processed/kb_batches/<batch_id>/`.
Local files và Qdrant không có transaction chung; giữ input/settings/snapshots để retry
batch pending. Không chạy writer này đồng thời với KB DAG trên cùng dữ liệu.

## 3. Model Training và artifacts

| Module | Mô hình |
|---|---|
| `src/absa/train_ml.py` | Logistic Regression, Random Forest |
| `src/absa/train_dl.py` | MultiHeadTextCNN, MultiHeadBiGRU |
| `src/absa/train_phobert.py` | Multi-head PhoBERT |
| `src/absa/evaluation.py` | Đánh giá model theo từng khía cạnh |

Các training script sử dụng W&B để log metrics, sweep cấu hình và lưu artifacts.
Dataset/checkpoint phải được cung cấp riêng; không có bước train tự động trong CI.

Kết quả **đã lưu trong evaluation JSON v2** (không phải benchmark vừa chạy):

| Model | Avg F1 macro | Artifact |
|---|---:|---|
| PhoBERT | 0.8476 | [phobert_evaluation.json](models/absa/v2/phobert/phobert_evaluation.json) |
| Logistic Regression | 0.7692 | [logreg_evaluation.json](models/absa/v2/logreg/logreg_evaluation.json) |
| Random Forest | 0.7208 | [random_forest_evaluation.json](models/absa/v2/random_forest/random_forest_evaluation.json) |
| TextCNN | 0.7434 | [cnn_evaluation.json](models/absa/v2/cnn/cnn_evaluation.json) |
| BiGRU | 0.7480 | [gru_evaluation.json](models/absa/v2/gru/gru_evaluation.json) |

Model files cần đặt đúng layout:

```text
models/
├── absa/v2/
│   ├── logreg/logreg.pkl
│   ├── phobert/phobert.pt
│   └── phobert_onnx/          # phobert_absa.onnx + tokenizer files
└── bge-m3-onnx/              # bge_m3_dense.onnx + tokenizer files, tùy chọn
```

`runtime/absa/paths.py` chọn `ABSA_VERSION` (v2 mặc định), ưu tiên layout trên;
cũng nhận `models/<model>/<file>` nếu checkpoint tương ứng tồn tại ở đó.

`scripts/convert_phobert_onnx.py` và `convert_bgem3_onnx.py` dùng đường dẫn
`/opt/airflow/models/...`; kiểm tra môi trường/path trước khi chạy export.
Không suy ra kích thước model hoặc mức tăng tốc nếu chưa đo artifact cụ thể.

## 4. Serving flow

![Serving Flow](docs/images/serving_flow.jpg)

### 4.1. HTTP chatbot

Flow của `runtime/chatbot/`:

```text
Chainlit → api_client.py → FastAPI
  ├─ query/history → rewrite khi cần → BGE-M3 → Qdrant dense search
  │                → metadata filters / ABSA rerank → LLM → SSE answer + sources
  └─ URL → /analyze → optional Redis → KB hoặc Tiki crawl → ABSA report + context_id
```

- UI entry: `runtime/chatbot/demo_chainlit.py`; API entry: `runtime.chatbot.api:app`.
- API chạy **một worker**; URL contexts giữ trong RAM, TTL một giờ, tối đa 128 mục.
- History do client gửi lại, tối đa 8 messages; API không nhận `session_id` cho luồng này.
- `LLM_PROVIDER=google|groq|auto`; auto thử Groq trước Google khi có key tương ứng.
- Retrieval dùng dense vectors; có sparse schema không đồng nghĩa serving chạy hybrid/BM25.
- Redis tùy chọn cho cache; warm-up tối đa 20 tên sản phẩm khi bật Redis/API_WARM_CACHE.
- Interaction logging bật qua env, mặc định runtime Compose ghi `logs/interactions.jsonl`.
- SSE có thể phát `error` sau HTTP 200; client chỉ lưu answer hoàn tất khi nhận `done`.

| Endpoint runtime | Input/chức năng |
|---|---|
| `GET /health` | Collection status/provider configuration; không xác minh LLM quota |
| `POST /search` | query, top_k, filters → documents/sources; không gọi LLM |
| `POST /chat` | Query/history/context_id → answer/sources |
| `POST /chat/stream` | Streaming sources/token/done/error |
| `POST /analyze` | url, absa_model → report/context_id |
| `GET /metrics` | Prometheus exposition khi metrics bật |

`top_k` từ 1–8. Filters: min_rating, price_min, price_max, doc_types, aspect_preferences.
URL analysis không tự upsert KB. Positive ratio dùng positive/(positive+negative),
không đại diện tỷ lệ hài lòng trên toàn bộ reviews.

### 4.2. Các entry point trong src/chatbot

`src/chatbot/demo_chainlit.py` gọi trực tiếp retrieval/LLM và giữ history trong phiên UI.
`src/chatbot/api.py` expose `/chat`, `/search`, `/health`, `/metrics`; Dockerfile của entry
này cấu hình hai worker. `/search` tại đây chỉ trả sources; không quản lý hội thoại bằng session_id.
LLM trong `src/chatbot/llm.py` dùng Groq/OpenRouter.

Các module này được chạy bởi `docker-compose.yml`; không nằm trong HTTP flow ở mục 4.1.

## 5. Monitoring, tracking và evaluation

### 5.1. Prometheus / Grafana

`docker-compose.runtime.yml` mount `monitoring/prometheus.yml` và Grafana provisioning:

- Scrape 5 giây, retention 15 ngày, dashboard 14 panel.
- Target `tiki-api` cho API; `tiki-loadtest` cho fixture.
- HTTP latency gồm toàn bộ response body; theo dõi thêm TTFT và stream outcomes.
- Bốn alert rules: API down, 5xx ratio, stream failure ratio và HTTP p95.
- Alert rules hiển thị trong Prometheus; chưa cấu hình notification channel.

`prometheus.yml` ở repository root có scrape interval 15 giây và được mount bởi
`docker-compose.yml`. Grafana provisioning trong `monitoring/` được gắn vào runtime Compose.

### 5.2. MLflow / W&B

W&B được gọi trong training và task retrain monitoring. MLflow có các experiment:

| Experiment | Nơi ghi |
|---|---|
| `retrain_monitoring` | `airflow/dags/absa_inference_dag.py` |
| `kb_update` | `airflow/dags/kb_rebuild_dag.py` |
| `ragas_evaluation` | `scripts/ragas_eval.py` |

Các integrations cần endpoint/credentials và dữ liệu phù hợp; runtime API không tự ghi
mọi request hoặc batch vào MLflow/W&B.

### 5.3. Ragas

`scripts/ragas_eval.py` dùng `src.chatbot.retrieval` và Groq để chuẩn bị question/answer/contexts,
chấm **faithfulness** và **answer relevancy**, sau đó log metrics/per-sample JSON vào MLflow.
Script lấy log có contexts hoặc dùng 20 fallback queries; mẫu lỗi/không có contexts có thể bị bỏ qua.

Script sử dụng dependency Ragas 0.1.21 trong môi trường API của `docker-compose.yml`.
Nó không tự gọi HTTP API runtime để đánh giá và không chạy trong CI.
Không công bố điểm Ragas cố định khi chưa có report của lần đo tương ứng.

### 5.4. Locust và CI

`tests/load_test.py`: HealthUser, SearchUser, ChatUser, StreamingUser, MixedUser.
`load_test_reference.py` là workload khác được lưu trong tests; Compose chọn `load_test.py`.
Runtime load-test profile mặc định gọi synthetic fixture và chặn LLM thật.

CI chạy khi push/PR/manual dispatch trên GitHub-hosted runner:

```text
Python 3.11 / 3.12 → pip check → static checks → 96 offline tests
Docker job → actionlint / Compose / promtool → UI + fixture build → HTTP/SSE/metrics smoke
```

Artifacts giữ 7 ngày. Workflow không deploy, không build/test toàn bộ ML production image
hoặc chạy DAG/S3/model thật. Repo chưa có CD workflow.

## 6. Cài đặt và vận hành

### 6.1. Chuẩn bị

- Docker Linux containers và Compose; Python 3.11/3.12 cho setup/tests.
- Checkpoint/tokenizer, dữ liệu canonical và KB snapshot hoặc documents để index.
- Provider key khi cần generation; credentials S3/W&B khi chạy các integration tương ứng.
- RAM/disk đủ cho model và các service đã chọn; không có capacity tối thiểu được benchmark trong repo.

```bash
git clone https://github.com/KhoiDang-KHDL/Tiki-Sentiment-Chatbot-RAG.git
cd Tiki-Sentiment-Chatbot-RAG
python scripts/setup_runtime.py
```

Script tạo `.env.runtime` khi chưa có, sinh Grafana password ngẫu nhiên và tạo thư mục output.
Cấu hình `LLM_PROVIDER`, keys, collection, ports, ABSA_VERSION và HF_HUB_OFFLINE trong file này.
Secrets, data, weights và reports sinh ra không được commit.

### 6.2. HTTP runtime

```bash
docker compose --env-file .env.runtime -f docker-compose.runtime.yml build api ui
docker compose --env-file .env.runtime -f docker-compose.runtime.yml up -d qdrant
```

Qdrant dùng volume riêng của project `tiki-rag-runtime`. Restore snapshot phù hợp hoặc
chuẩn bị `data/processed/documents.jsonl` rồi index (không cần `--recreate`):

```bash
docker compose --env-file .env.runtime -f docker-compose.runtime.yml run --rm --no-deps \
  -v "${PWD}/data:/app/data" api python -m runtime.knowledge_base.index_qdrant
```

Collection phải tồn tại trước khi start API:

```bash
docker compose --env-file .env.runtime -f docker-compose.runtime.yml up -d --wait
docker compose --env-file .env.runtime -f docker-compose.runtime.yml ps
docker compose --env-file .env.runtime -f docker-compose.runtime.yml exec -T ui python -m runtime.chatbot.container_check
```

PowerShell wrapper: `scripts/runtime.ps1` hỗ trợ config/build/start/stop/status/check/locust/password.
Chi tiết: [docs/RUNTIME.md](docs/RUNTIME.md).

### 6.3. Airflow / S3 services

`docker-compose.yml` đọc `.env`, build image Airflow qua `Dockerfile.airflow` và mount src/data/models.
Chuẩn bị `.env` từ `.env.example`; cấu hình bucket, region, artifact URI và dependencies trước khi bật DAG.
Scheduler/worker dùng image `tiki-airflow-custom:latest` được build từ service airflow.
Không chạy song song KB DAG và scripts/update_kb.py trên cùng collection/canonical data.

## 7. Kiểm thử

Offline tests, không cần model/provider key:

```bash
python -m venv .venv-ci
source .venv-ci/bin/activate
python -m pip install -r requirements-ci.txt
python -m scripts.ci_checks
python -m scripts.ci_tests
```

Request runtime qua API; history/context_id được gửi rõ ràng:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query":"sản phẩm chất lượng tốt","top_k":3}'
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"url":"https://tiki.vn/product-p53497035.html","absa_model":"logreg"}'
# Lấy context_id từ response trên trước khi hỏi tiếp.
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query":"Có nên mua sản phẩm này?","history":[],"context_id":"CONTEXT_ID_FROM_ANALYZE","focus_current":true}'
```

Thay context_id bằng kết quả `/analyze`; chat dùng provider thật và có thể tốn quota.
Locust fixture UI:

```bash
docker compose --env-file .env.runtime -f docker-compose.runtime.yml --profile loadtest up -d api-loadtest locust
```

Report sinh tại `reports/loadtest/`. Không suy ra giới hạn 20 users, provider RPM hoặc latency
cố định nếu chưa đo đúng môi trường. Xem [monitoring/README.md](monitoring/README.md).

## 8. Truy cập services

| Service | Port mặc định | Cấu hình |
|---|---:|---|
| HTTP API / docs | 8000 /docs | .env.runtime |
| Chainlit | 8001 | .env.runtime |
| Qdrant | 6333 | .env.runtime |
| Grafana | 3000 | .env.runtime; credentials sinh bởi setup |
| Prometheus | 9090 | .env.runtime |
| Locust | 8089 | Profile loadtest |
| Airflow UI | 8080 | docker-compose.yml |
| MLflow | 5000 | docker-compose.yml |
| Redis | 6379 | docker-compose.yml |

Runtime host ports bind 127.0.0.1. Airflow/PostgreSQL/Redis/MLflow không được start bởi runtime Compose.
Không dùng admin/admin cho Grafana runtime; lấy password từ `.env.runtime` hoặc wrapper password.

## 9. Cấu trúc thư mục

```text
Tiki-Sentiment-Chatbot-RAG/
├── src/
│   ├── absa/                   # Dataset, train, evaluation, inference
│   ├── chatbot/                # Direct UI, API, retrieval, Groq/OpenRouter
│   ├── crawling/               # Bulk/incremental crawl, S3 upload
│   ├── preprocessing/          # Clean detail/review
│   └── kb/                     # Aggregate, build, full/delta indexing
├── runtime/
│   ├── chatbot/                # HTTP UI/API, SSE, Google/Groq, metrics
│   ├── absa/                   # Inference và checkpoint path resolver
│   ├── crawling/               # Shared client và URL analysis crawler
│   ├── preprocessing/          # Normalization cho runtime
│   └── knowledge_base/         # Resumable incremental writer
├── airflow/dags/               # Năm DAG độc lập theo lịch
├── scripts/                    # Export, Ragas, setup, incremental và CI
├── tests/                      # Regression, SSE parser, checkpoint tests, Locust
├── s3_lambda/                  # Labeling/S3 integration scripts
├── models/                     # Evaluation JSON; weights được cấp riêng
├── monitoring/                 # Metrics config, alert rules, Grafana provisioning
├── docker/                     # API/UI/runtime/CI Dockerfiles
├── docs/images/                # Các hình minh họa kiến trúc
├── .github/workflows/ci.yml    # CI, không deployment
├── docker-compose.yml          # Các service src/Airflow/S3/MLflow
└── docker-compose.runtime.yml  # HTTP runtime, Qdrant, monitoring, load-test profile
```

## 10. Phạm vi vận hành

- Dữ liệu và model weights không đi kèm repository; point count phụ thuộc collection được cấp.
- ABSA evaluation JSON không thay thế phép đo model/runtime/KB đang triển khai.
- Live crawl và generation phụ thuộc Tiki API, provider model availability và quota.
- Runtime API chưa có auth/rate limiting; không mở public trực tiếp.
- S3/W&B/DAG integrations cần cấu hình và kiểm tra end-to-end riêng; CI không chứng minh các service đó hoạt động.
- Không có cấu hình EC2/CloudWatch hoặc workflow tự động triển khai trong repo.
