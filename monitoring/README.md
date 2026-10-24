# Runtime monitoring

Áp dụng `docker-compose.runtime.yml`; không thay config của stack tham khảo.

```powershell
.\scripts\runtime.ps1 start
.\scripts\runtime.ps1 check
.\scripts\runtime.ps1 password
.\scripts\runtime.ps1 locust
```

| Service | URL mặc định |
|---|---|
| Grafana | http://localhost:3000/d/tiki-api |
| Prometheus targets / alerts | http://localhost:9090/targets / http://localhost:9090/alerts |
| Locust | http://localhost:8089 |

Dashboard 14 panel, Target `tiki-api` cho runtime thật, `tiki-loadtest` cho fixture.
Scrape 5 giây, retention 15 ngày. HTTP latency gồm toàn bộ response/SSE body.
HTTP 200 có thể chứa SSE error; đọc thêm stream outcomes và TTFT.
CPU/RAM là của tiến trình API; metrics không chứa query/product ID labels.
Bốn alert rules hiển thị trong Prometheus, chưa có notification channel.

Locust UI: chọn `MixedUser`, host `http://api-loadtest:8000`, cấu hình users/spawn rate rồi Start.
Retrieval thật: chọn `SearchUser`, host `http://api:8000`. Live LLM bị chặn mặc định.
Headless fixture test:

```powershell
docker compose --env-file .env.runtime -f docker-compose.runtime.yml --profile loadtest run --rm -e LOADTEST_MODE=synthetic -e LOADTEST_SCENARIO=mixed -e LOADTEST_SUMMARY=/reports/summary.json locust -f /mnt/locust/load_test.py --headless --host http://api-loadtest:8000 -u 5 -r 1 -t 30s --html /reports/report.html MixedUser
```

Output tại `reports/loadtest/`: summary.json, requests.csv, report.html.
Exit 1 khi failure ratio >1%, p95 >10s hoặc không có requests. SSE error/missing done/empty
completion là failure. TTFT samples chỉ gồm stream thành công; đọc kèm failure count.
Đây là closed-loop workload với wait 0,3–0,8s, chưa phải capacity benchmark.
Không coi kết quả fixture là tốc độ model/LLM thật.
