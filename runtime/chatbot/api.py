"""Local API: python -m uvicorn runtime.chatbot.api:app --host 127.0.0.1 --port 8000.

Retrieval is loaded once during lifespan, not at import time. History is supplied
by callers; there is no shared conversation state between users.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import re
import secrets
from pathlib import Path
from threading import Lock
from time import perf_counter, monotonic
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=False)

# Load environment before retrieval's module-level configuration is evaluated.
from runtime.chatbot.retrieval import ASPECTS, COLLECTION_NAME, TikiRAG, RagFilters
from runtime.chatbot import llm
from runtime.chatbot import metrics as monitoring

logger = logging.getLogger(__name__)
_log_lock = Lock()


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=8, strict=True)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    doc_types: list[Literal["product_card", "aspect_summary", "review"]] = Field(default_factory=list, max_length=3)
    aspect_preferences: dict[str, float] = Field(default_factory=dict, max_length=6)

    @model_validator(mode="after")
    def validate_filters(self):
        if self.price_min is not None and self.price_max is not None and self.price_min > self.price_max:
            raise ValueError("price_min must not exceed price_max")
        for aspect, threshold in self.aspect_preferences.items():
            if aspect not in ASPECTS or not -1 <= threshold <= 1:
                raise ValueError("Aspect preferences require known aspects and scores between -1 and 1")
        return self


class ChatRequest(SearchRequest):
    history: list[HistoryMessage] = Field(default_factory=list, max_length=8)
    context_id: str | None = Field(default=None, min_length=1, max_length=128)
    focus_current: bool = False


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    url: str = Field(min_length=1, max_length=2048)
    absa_model: Literal["logreg", "phobert_onnx", "phobert"] = "logreg"


class Source(BaseModel):
    id: str
    doc_type: str | None = None
    name: str = ""
    score: float
    product_id: int | None = None
    product_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchResponse(BaseModel):
    sources: list[Source]
    documents: list[dict]


def _warm_cache(rag):
    """Optional, bounded cache warming using the configured collection."""
    if os.getenv("API_WARM_CACHE", "false").lower() != "true":
        return
    if rag.cache is None:
        logger.warning("Skipping cache warm-up: Redis is not enabled")
        return
    from qdrant_client.http import models as qm
    try:
        points, _ = rag.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key="doc_type", match=qm.MatchValue(value="product_card"))]),
            limit=20, with_payload=True, with_vectors=False)
        names = sorted({(p.payload or {}).get("metadata", {}).get("name", "") for p in points})
        for name in names:
            if name:
                rag.search(name[:200], top_k=1)
    except Exception as exc:
        logger.warning("Cache warm-up failed (%s)", type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    app.state.retrieval_lock = Lock()
    app.state.llm_lock = Lock()
    app.state.context_lock = Lock()
    app.state.product_contexts = {}
    try:
        app.state.rag = await run_in_threadpool(TikiRAG)
        await run_in_threadpool(_warm_cache, app.state.rag)
        yield
    finally:
        try:
            if app.state.rag is not None:
                await run_in_threadpool(app.state.rag.close)
        finally:
            await run_in_threadpool(llm.close_clients)
            app.state.rag = None


app = FastAPI(title="Tiki RAG Chatbot", lifespan=lifespan)
app.add_middleware(monitoring.MonitoringMiddleware)
if os.getenv("API_ENABLE_METRICS", "false").lower() == "true":
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)


def _retrieve(request: Request, req: SearchRequest, query: str | None = None):
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(503, "Retrieval has not finished starting")
    filters = RagFilters(doc_types=req.doc_types, min_rating=req.min_rating,
                         price_min=req.price_min, price_max=req.price_max,
                         aspect_preferences=req.aspect_preferences)
    try:
        # FlagEmbedding shares a single CPU model across request threads.
        with request.app.state.retrieval_lock:
            return rag.search(query or req.query, top_k=req.top_k, filters=filters)
    except Exception as exc:
        logger.warning("Retrieval failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Retrieval unavailable; check Qdrant and the embedding model") from None


def _sources(docs):
    result = []
    for doc in docs:
        metadata = doc.get("metadata") or {}
        result.append(Source(id=str(doc["id"]), doc_type=doc.get("doc_type"),
                             name=metadata.get("name") or "", score=round(float(doc["score"]), 3),
                             product_id=metadata.get("product_id"), product_url=metadata.get("product_url")))
    return result


def _log_interaction(question, answer, docs, latency_ms):
    # Logging is explicit; paths are local to the project rather than /app.
    if os.getenv("API_LOG_INTERACTIONS", "false").lower() != "true":
        return
    path = Path(os.getenv("API_LOG_FILE", "logs/interactions.jsonl"))
    if not path.is_absolute():
        path = BASE_DIR / path
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "question": question,
             "answer": answer, "sources": [source.model_dump() for source in _sources(docs)],
             "latency_ms": latency_ms, "n_docs": len(docs)}
    try:
        with _log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    except OSError as exc:
        logger.warning("Interaction logging failed (%s)", type(exc).__name__)


def _llm_error(exc):
    status = 503 if isinstance(exc, llm.LLMConfigurationError) else 502
    return HTTPException(status, str(exc))


def _chat_documents(request, req):
    history = [turn.model_dump() for turn in req.history]
    current = None
    if req.context_id:
        with request.app.state.context_lock:
            stored = request.app.state.product_contexts.get(req.context_id)
            if stored is None or stored["expires"] < monotonic():
                request.app.state.product_contexts.pop(req.context_id, None)
                raise HTTPException(409, "Ngữ cảnh sản phẩm đã hết hạn; gửi lại URL hoặc dùng /clear.")
            current = stored["document"]
    lowered = req.query.lower()
    alternative = any(word in lowered for word in ("thay thế", "sản phẩm khác", "sp khác", "tương tự", "cùng phân khúc", "so sánh với"))
    followup = any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", lowered)
                   for word in ("sản phẩm này", "sp này", "món này", "cái này", "hàng này", "nó", "link vừa gửi", "vừa gửi"))
    generic = lowered.strip(" .?!") in ("có nên mua", "có nên mua không", "đáng mua không", "ổn không", "review đi",
        "ưu điểm là gì", "nhược điểm là gì", "chất lượng thế nào", "giao hàng thì sao")
    if current and not alternative and (req.focus_current or followup or generic):
        return [current], history
    if current and alternative:
        metadata = current["metadata"]
        query = f"{req.query}. Thay thế cho {metadata.get('name', '')}. Danh mục: {metadata.get('category_name', '')}"
        candidates = _retrieve(request, req, query[:4000])
        others = [doc for doc in candidates if str((doc.get("metadata") or {}).get("product_id")) != str(metadata["product_id"])]
        return [current, *others[:7]], history
    # Query rewriting may call an LLM; only conversation requests need it.
    with request.app.state.llm_lock:
        query = llm.rewrite_query(req.query, history)
    return _retrieve(request, req, query), history


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, request: Request):
    from runtime.chatbot.url_analyzer import (
        AnalysisError, analyze_product, analysis_to_document, format_analysis_report, extract_product_id,
    )
    if extract_product_id(req.url) is None:
        raise HTTPException(422, "URL Tiki hoặc product ID không hợp lệ")
    try:
        analysis = await analyze_product(req.url, absa_model=req.absa_model)
    except (AnalysisError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from None
    except Exception as exc:
        logger.warning("URL analysis failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Phân tích URL chưa khả dụng; kiểm tra Tiki và model ABSA") from None
    doc = analysis_to_document(analysis)
    context_id = secrets.token_urlsafe(32)
    with request.app.state.context_lock:
        contexts = request.app.state.product_contexts
        expired = [key for key, value in contexts.items() if value["expires"] < monotonic()]
        for key in expired:
            del contexts[key]
        while len(contexts) >= 128:
            contexts.pop(next(iter(contexts)))
        contexts[context_id] = {"document": doc, "expires": monotonic() + 3600}
    return {"context_id": context_id, "product_id": analysis.product_id,
            "report": format_analysis_report(analysis), "sources": [source.model_dump() for source in _sources([doc])]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    start = perf_counter()
    try:
        docs, history = _chat_documents(request, req)
        with request.app.state.llm_lock:
            answer = llm.ask(req.query, docs, history)
    except llm.LLMError as exc:
        raise _llm_error(exc) from None
    _log_interaction(req.query, answer, docs, round((perf_counter() - start) * 1000))
    return ChatResponse(answer=answer, sources=_sources(docs))


@app.post("/search", response_model=SearchResponse)
def search_only(req: SearchRequest, request: Request):
    docs = _retrieve(request, req)
    return SearchResponse(sources=_sources(docs), documents=docs)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    start = perf_counter()
    try:
        docs, history = _chat_documents(request, req)
        if docs:
            llm.configured_providers()
    except llm.LLMError as exc:
        monitoring.stream_preflight_error(start)
        raise _llm_error(exc) from None
    except Exception:
        monitoring.stream_preflight_error(start)
        raise

    def event(name, data):
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        observation = monitoring.StreamObservation(start)
        outcome = 'cancelled'
        pieces = []
        iterator = None
        try:
            yield event("sources", [source.model_dump() for source in _sources(docs)])
            with request.app.state.llm_lock:
                iterator = llm.ask_stream(req.query, docs, history)
                for text in iterator:
                    observation.token(text)
                    pieces.append(text)
                    yield event("token", {"text": text})
            _log_interaction(req.query, "".join(pieces), docs, round((perf_counter() - start) * 1000))
            yield event("done", {"status": "ok"})
            outcome = 'completed'
        except llm.LLMError as exc:
            outcome = 'provider_error'
            # Headers are already sent; clients must handle the SSE error event.
            yield event("error", {"detail": str(exc)})
            return
        except Exception as exc:
            outcome = 'internal_error'
            logger.warning("Streaming failed (%s)", type(exc).__name__)
            yield event("error", {"detail": "LLM stream interrupted"})
            return
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
            finally:
                observation.finish(outcome)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
def health(request: Request):
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(503, "Retrieval not initialized")
    try:
        with request.app.state.retrieval_lock:
            info = rag.client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Health check failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Qdrant collection unavailable") from None
    try:
        providers = llm.configured_providers()
    except llm.LLMConfigurationError:
        providers = []
    return {"status": "ok", "collection": COLLECTION_NAME, "points_count": info.points_count,
            "llm_configured": bool(providers), "llm_providers": providers}
