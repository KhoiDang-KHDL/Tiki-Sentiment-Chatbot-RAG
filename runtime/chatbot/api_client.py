"""HTTP-only UI transport: no models, Qdrant clients or provider API keys."""
import json
import os
from urllib.parse import urlparse
import httpx


class APIError(RuntimeError):
    pass


def base_url():
    value = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise APIError("API_BASE_URL phải là URL HTTP(S) không chứa credentials")
    return value


async def check_response(response):
    if response.is_success:
        return
    await response.aread()
    try:
        detail = response.json().get("detail")
        if not isinstance(detail, str):
            detail = f"API trả HTTP {response.status_code}; kiểm tra dữ liệu gửi lên"
    except (ValueError, AttributeError):
        detail = f"API trả HTTP {response.status_code}"
    raise APIError(detail[:1200])


async def analyze(url, absa_model="logreg"):
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            response = await client.post(base_url() + "/analyze", json={"url": url, "absa_model": absa_model})
            await check_response(response)
            return response.json()
    except httpx.HTTPError as exc:
        raise APIError("Không kết nối được API; kiểm tra service API và API_BASE_URL") from exc


async def parse_events(lines):
    name, data = "message", []
    async for line in lines:
        if not line:
            if data:
                try:
                    yield name, json.loads("\n".join(data))
                except ValueError as exc:
                    raise APIError("API trả stream không hợp lệ") from exc
            name, data = "message", []
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


async def stream_chat(query, history, *, context_id=None, focus_current=False):
    payload = {"query": query, "history": history[-8:], "top_k": 8 if context_id else 5,
               "context_id": context_id, "focus_current": focus_current}
    done = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            async with client.stream("POST", base_url() + "/chat/stream", json=payload) as response:
                await check_response(response)
                async for name, data in parse_events(response.aiter_lines()):
                    if name == "error":
                        raise APIError(str(data.get("detail", "Stream bị gián đoạn")))
                    if name == "done":
                        done = True
                    yield name, data
        if not done:
            raise APIError("Kết nối đã đóng trước khi API hoàn tất câu trả lời")
    except httpx.HTTPError as exc:
        raise APIError("Kết nối API bị gián đoạn; hãy thử lại") from exc
