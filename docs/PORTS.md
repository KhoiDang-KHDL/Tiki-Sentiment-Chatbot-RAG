# Integration map

| Thành phần Tiki_ABSA | Đích trong repo này | Điều chỉnh |
|---|---|---|
| `source/chatbot/` | `runtime/chatbot/` | Namespace, HTTP UI/API và Google/Groq |
| `source/absa/` | `runtime/absa/` | Resolver ưu tiên versioned upstream checkpoint, fallback flat checkpoint |
| `source/crawling/`, `source/preprocessing/` | `runtime/` tương ứng | Dùng riêng, không ghi đè crawler/training gốc |
| `source/knowledge_base/`, `scripts/update_kb.py` | `runtime/knowledge_base/`, `scripts/update_kb.py` | Model revision fingerprint theo đúng checkpoint được chọn |
| Metrics + provisioning | `monitoring/`, `runtime/chatbot/metrics.py` | Target thật/fixture tách riêng |
| Offline tests | `tests/test_*.py` | Imports sang runtime; thêm checkpoint path tests |
| Locust | `tests/load_test.py` | Synthetic default, live LLM gate; giữ workload gốc ở load_test_reference.py |
| CI | `.github/workflows/ci.yml` | Python matrix, static checks, Compose/promtool và container smoke |

## Giữ riêng

- `src/` vẫn là implementation tham khảo; các import không tự chuyển sang runtime.
- Original `docker-compose.yml` vẫn phục vụ stack Airflow/S3/MLflow cũ.
- `docker-compose.runtime.yml` có project/network/volumes riêng, không include Airflow.
- Không port Airflow 3 LocalExecutor/DAG từ Tiki_ABSA vào Airflow 2 Celery stack.
- Không port secrets, checkpoint weights, data, generated reports hoặc deploy targets của máy cũ.

## Các thay đổi hygiene

- Loại bảng thành viên, môn học và EC2 khỏi README.
- Loại automatic deploy workflow trỏ school server và EC2.
- Tiki guest token literal trong crawler gốc được thay bằng `TIKI_GUEST_TOKEN` environment variable.
- Notebook outputs được strip trước khi commit; giữ code cells.
- `.gitignore` và `.dockerignore` loại private env, cache, data và model weights.

CI kiểm tra namespace runtime và compile source tham khảo. Chưa chạy original training,
S3/Lambda/DAG integration hay model inference thật trên repo mới.
