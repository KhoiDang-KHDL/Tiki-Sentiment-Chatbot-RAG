from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
"""
FastAPI endpoint — POST /chat, /search
Pre-warm cache from KB data on startup
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()

from src.chatbot.retrieval import TikiRAG, RagFilters
from src.chatbot import llm

app = FastAPI(title="Tiki RAG Chatbot")
Instrumentator().instrument(app).expose(app)
rag = TikiRAG()


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    min_rating: Optional[float] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.on_event("startup")
def warm_cache():
    """Pre-warm Redis cache with queries from actual KB data."""
    import threading

    def _warm():
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm

            client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

            results = client.scroll(
                collection_name="tiki_kb",
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="doc_type", match=qm.MatchValue(value="product_card"))
                ]),
                limit=100,
                with_payload=True,
            )[0]

            brands = set()
            categories = set()
            names = set()
            for r in results:
                m = (r.payload or {}).get("metadata", {})
                if m.get("brand_name"):
                    brands.add(m["brand_name"])
                if m.get("category_name"):
                    categories.add(m["category_name"])
                if m.get("name"):
                    names.add(m["name"][:60])

            queries = set()

            for b in brands:
                queries.add(f"sản phẩm {b} tốt nhất")
                queries.add(f"{b} giá rẻ")

            for c in categories:
                if c and c != "Root":
                    queries.add(f"{c} chất lượng tốt")
                    queries.add(f"{c} giá rẻ")
                    queries.add(f"{c} được đánh giá cao")

            for n in list(names)[:20]:
                queries.add(n)

            common = [
                "cốc giữ nhiệt tốt",
                "bình giữ nhiệt Lock&Lock",
                "đồ gia dụng dưới 300k",
                "sản phẩm giao hàng nhanh",
                "chảo chống dính tốt nhất",
                "pin tiểu chính hãng",
                "bình nước inox",
                "nồi chiên không dầu",
                "sản phẩm chất lượng cao",
                "sản phẩm được đánh giá tốt",
                "sản phẩm dưới 100k",
                "sản phẩm dưới 200k",
                "sản phẩm dưới 500k",
            ]
            queries.update(common)

            count = 0
            for q in queries:
                try:
                    rag.search(q, top_k=1)
                    count += 1
                except Exception:
                    pass

            print(f"Cache warmed: {count}/{len(queries)} queries (brands={len(brands)}, categories={len(categories)}, products={len(names)})")

        except Exception as e:
            print(f"Cache warm failed: {e}")

    thread = __import__("threading").Thread(target=_warm, daemon=True)
    thread.start()
    print("Cache warming started in background...")



import json as _json, time as _time, pathlib as _pathlib

_LOG_FILE = _pathlib.Path("/app/logs/interactions.jsonl")
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _log_interaction(question, answer, docs, latency_ms):
    try:
        entry = {
            "timestamp":  __import__("datetime").datetime.utcnow().isoformat(),
            "question":   question,
            "answer":     answer,
            "contexts":   [d.get("text", "")[:500] for d in docs],
            "sources":    [{"doc_type": d["doc_type"], "name": d["metadata"].get("name",""), "score": round(d["score"],3)} for d in docs],
            "latency_ms": latency_ms,
            "n_docs":     len(docs),
        }
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log] failed: {e}")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _t0 = _time.time()
    filters = RagFilters(
        min_rating=req.min_rating,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    docs   = rag.search(req.query, top_k=req.top_k, filters=filters)
    answer = llm.ask(req.query, docs)
    sources = [{"doc_type": d["doc_type"], "name": d["metadata"].get("name", ""), "score": round(d["score"], 3)} for d in docs]
    _log_interaction(req.query, answer, docs, round((_time.time() - _t0) * 1000))
    return ChatResponse(answer=answer, sources=sources)


@app.post("/search")
def search_only(req: ChatRequest):
    filters = RagFilters(
        min_rating=req.min_rating,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    docs = rag.search(req.query, top_k=req.top_k, filters=filters)
    sources = [{"doc_type": d["doc_type"], "name": d["metadata"].get("name", ""), "score": round(d["score"], 3)} for d in docs]
    return {"sources": sources}


@app.get("/health")
def health():
    return {"status": "ok"}
