# Optional hardened runtime

Runtime được port từ Tiki_ABSA, giữ `src/` tham khảo độc lập.
Chạy các lệnh từ repository root. Hai stack không dùng chung image/volume theo mặc định.

## Configuration

```powershell
py -3.12 -m scripts.setup_runtime
```

Script tạo `.env.runtime` khi chưa có và sinh Grafana password ngẫu nhiên.
Sửa provider credentials, collection và ports trước khi start. Không commit env thật.

| Biến | Ý nghĩa |
|---|---|
| `LLM_PROVIDER` | google/groq/auto; auto thử Groq trước Google |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY` | Provider keys |
| `QDRANT_COLLECTION` | Collection runtime, mặc định tiki_kb |
| `ABSA_VERSION` | v2 mặc định; hỗ trợ v1 |
| `HF_HUB_OFFLINE` | 0 cho phép tải public model; 1 yêu cầu cache đủ |
| `API_HOST_PORT`, `UI_HOST_PORT`, `QDRANT_HOST_PORT` | Host ports |
| `GRAFANA_ADMIN_PASSWORD` | Sinh bởi setup; chỉ dùng khi Grafana DB khởi tạo |

Model resolver ưu tiên `models/absa/<version>/<model>/<file>` rồi fallback
`models/<model>/<file>`. PhoBERT ONNX cần tokenizer trong cùng model directory.
BGE-M3 mặc định dùng FlagEmbedding; `models/bge-m3-onnx/` là tùy chọn.
Runtime ML image pin scikit-learn 1.6.1; cần kiểm tra checkpoint được cấp tương thích.

## Build và chuẩn bị KB

```powershell
.\scripts\runtime.ps1 build
docker compose --env-file .env.runtime -f docker-compose.runtime.yml up -d qdrant
```

Qdrant này dùng volume mới `tiki-rag-runtime_runtime_qdrant`; chưa có dữ liệu.
Restore snapshot collection phù hợp, hoặc mount canonical documents để full index:

```powershell
docker compose --env-file .env.runtime -f docker-compose.runtime.yml run --rm --no-deps -v "${PWD}/data:/app/data" api python -m runtime.knowledge_base.index_qdrant --help
```

Đọc options của indexer và kiểm tra đích collection trước khi chạy lệnh ghi.
Không dùng full index để sửa một batch pending. Không trỏ vào collection thật của
stack khác nếu chưa quyết định rõ quyền sở hữu writer.
API startup chỉ thành công sau khi collection đã tồn tại và model/cache sẵn sàng.

```powershell
.\scripts\runtime.ps1 start
.\scripts\runtime.ps1 check
.\scripts\runtime.ps1 status
.\scripts\runtime.ps1 stop
```

`check` gọi API health/search và Prometheus/Grafana; không gọi LLM.
`start` không build lại code. `stop` giữ volumes; không dùng `down -v` để bảo toàn dữ liệu.

## Incremental update

Dùng runtime pipeline thay thế writer cũ, không chạy song song Airflow KB DAGs.
Input UTF-8: `product_id,content,rating`, product đã có trong `data/raw/products_detail.csv`.
Canonical files trong `data/processed/`: labeled review CSV, aspect score parquet và documents.jsonl.
Cần các canonical artifact ban đầu; clone repo không cung cấp dataset.

Dry-run trong API image, với data writable và model read-only từ Compose:

```powershell
docker compose --env-file .env.runtime -f docker-compose.runtime.yml run --rm --no-deps -v "${PWD}/data:/app/data" api python -m scripts.update_kb --input data/raw/reviews_batch_001.csv --collection tiki_kb --model logreg --dry-run
```

Bỏ `--dry-run` để cập nhật thật. Qdrant service phải chạy. Với collection khác, thay
`--collection` tương ứng cấu hình. Chỉ một writer dùng canonical data tại một thời điểm.

- Dedup theo product_id và normalized content; append-only, chưa edit/delete review.
- Backup/snapshots/manifest/done tại `data/processed/kb_batches/<batch_id>/`.
- Stable IDs: upsert trước, xóa stale IDs sau và verify payload.
- Lỗi partial write: giữ pending marker/input/model/settings; retry đúng lệnh gốc.
- Không xóa pending/snapshots hoặc sửa canonical/detail khi batch chưa hoàn tất.
- Không có transaction chung hoặc rollback tự động giữa local files và Qdrant.
- Full builder gốc chưa có cùng direct-review policy; sau incremental giữ canonical documents export.

## Monitoring và load test

```powershell
.\scripts\runtime.ps1 password
.\scripts\runtime.ps1 locust
```

Grafana: localhost:3000/d/tiki-api; Prometheus: localhost:9090; Locust: localhost:8089.
Locust mặc định gọi synthetic fixture, không mount keys/models/data và không gọi LLM.
Port/state thay đổi theo `.env.runtime`; Grafana/Prometheus dùng volumes riêng.
Chi tiết: [monitoring](../monitoring/README.md).

## Verification boundary

Offline tests kiểm tra API/UI/SSE, crawler/cleaner, incremental retry và checkpoint resolver.
CI smoke dùng fixture. Không dùng kết quả đó để khẳng định checkpoint, KB thật, provider
hoặc original Airflow/S3 pipeline đã chạy thành công. Ragas script gốc vẫn đánh giá `src/`.
