"""
RAGAS Evaluation — Tiki Chatbot (domain: nhà cửa & đời sống)
=============================================================
Install (one-time):
  docker exec tiki-api pip install ragas==0.1.21 langchain-groq \
      langchain-community "sentence-transformers>=2.2.2" datasets mlflow

Run:
  docker exec tiki-api python /app/scripts/ragas_eval.py
"""
from __future__ import annotations
from typing import Optional, List

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://tiki-mlflow:5000")
QDRANT_URL   = os.getenv("QDRANT_URL",          "http://tiki-qdrant:6333")
COLLECTION   = "tiki_kb"
TOP_K        = 3
GROQ_MODEL   = "llama-3.3-70b-versatile"
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOG_FILE     = BASE / "logs" / "interactions.jsonl"
MIN_REAL_LOGS = 10

# ── Câu hỏi domain nhà cửa & đời sống ────────────────────────────────────────
FALLBACK_QUESTIONS = [
    # Bếp & nấu ăn
    "Nồi cơm điện nào nấu ngon và tiết kiệm điện nhất?",
    "Bình đun siêu tốc loại nào bền và an toàn nhất?",
    "Chảo chống dính nào tốt dùng được lâu?",
    "Máy xay sinh tố mini nào xay được đá không bị hỏng?",
    "Nồi chiên không dầu dung tích 5 lít nào tốt nhất?",
    # Đồ uống & giữ nhiệt
    "Cốc giữ nhiệt nào tốt nhất dưới 500k?",
    "Bình giữ nhiệt 1 lít nào giữ nhiệt lâu nhất?",
    "Ấm đun nước loại nào vừa nhanh vừa tiết kiệm điện?",
    # Làm sạch & vệ sinh
    "Robot hút bụi thông minh tầm 3 triệu có tốt không?",
    "Máy lau nhà nào lau sạch và dễ sử dụng?",
    "Nước tẩy rửa nhà bếp nào hiệu quả nhất?",
    # Không khí & nhiệt độ
    "Máy lọc không khí phù hợp cho phòng ngủ 20m2?",
    "Quạt tích điện dùng được bao nhiêu tiếng một lần sạc?",
    "Máy tạo độ ẩm nào phù hợp cho phòng ngủ trẻ em?",
    # Sức khỏe & chăm sóc cá nhân
    "Máy massage cầm tay nào xung lực mạnh mà giá hợp lý?",
    "Cân điện tử thông minh nào đo được chỉ số cơ thể?",
    "Máy lọc nước nano nào lọc sạch và ít tốn lõi lọc nhất?",
    # So sánh & tư vấn
    "Tủ lạnh mini nào tiêu thụ điện thấp phù hợp phòng trọ?",
    "Bếp từ đơn nào an toàn và dễ vệ sinh nhất?",
    "Máy sấy tóc nào ít gây hư tóc nhất?",
]


# ── Retrieval (dùng TikiRAG production) ───────────────────────────────────────
_rag = None


def get_rag():
    global _rag
    if _rag is None:
        logger.info("Loading TikiRAG...")
        from src.chatbot.retrieval import TikiRAG
        _rag = TikiRAG()
    return _rag


def get_contexts(question: str) -> List[str]:
    docs = get_rag().search(question, top_k=TOP_K)
    return [d.get("text", "").strip() for d in docs if d.get("text")]


# ── Generation ────────────────────────────────────────────────────────────────
def get_answer(question: str, contexts: List[str]) -> str:
    from groq import Groq
    ctx_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    client  = Groq(api_key=GROQ_API_KEY)
    resp    = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý tư vấn sản phẩm Tiki. "
                    "Chỉ dùng thông tin được cung cấp để trả lời. "
                    "Trả lời ngắn gọn, rõ ràng bằng tiếng Việt."
                ),
            },
            {
                "role": "user",
                "content": f"Thông tin sản phẩm:\n{ctx_str}\n\nCâu hỏi: {question}",
            },
        ],
        max_tokens=400,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# ── Load từ real logs ─────────────────────────────────────────────────────────
def load_from_logs() -> tuple:
    if not LOG_FILE.exists():
        logger.info("interactions.jsonl not found — dùng fallback questions")
        return [], [], []
    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    if len(entries) < MIN_REAL_LOGS:
        logger.info(f"Chỉ có {len(entries)} real logs — dùng fallback questions")
        return [], [], []
    logger.info(f"Loaded {len(entries)} real queries")
    valid = [
        (e["question"], e["answer"], e.get("contexts", []))
        for e in entries if e.get("contexts")
    ]
    if not valid:
        return [], [], []
    q, a, c = zip(*valid)
    return list(q), list(a), list(c)


# ── Build từ fallback ─────────────────────────────────────────────────────────
def build_from_fallback() -> tuple:
    logger.info(f"Building dataset từ {len(FALLBACK_QUESTIONS)} fallback questions...")
    all_q, all_a, all_c = [], [], []
    for i, q in enumerate(FALLBACK_QUESTIONS, 1):
        logger.info(f"[{i:02d}/{len(FALLBACK_QUESTIONS)}] {q}")
        try:
            contexts = get_contexts(q)
            if not contexts:
                logger.warning("  → no contexts, skipping")
                continue
            answer = get_answer(q, contexts)
            all_q.append(q)
            all_a.append(answer)
            all_c.append(contexts)
            logger.info(f"  → {len(contexts)} docs | {answer[:65]}...")
            time.sleep(0.3)
        except Exception as exc:
            logger.warning(f"  → error: {exc}, skipping")
    return all_q, all_a, all_c


# ── RAGAS ─────────────────────────────────────────────────────────────────────
def run_ragas(questions: List, answers: List, contexts: List) -> dict:
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
    )

    faithfulness.llm            = llm
    answer_relevancy.llm        = llm
    answer_relevancy.embeddings = embeddings

    dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    logger.info(f"Running RAGAS trên {len(questions)} samples (ETA: 5-8 min)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    return {
        "faithfulness":     float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "n_samples":        len(df),
        "df":               df,
    }


# ── MLflow ────────────────────────────────────────────────────────────────────
def log_mlflow(scores: dict, source: str) -> None:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ragas_evaluation")
    run_name = f"ragas_{source}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "groq_model":  GROQ_MODEL,
            "top_k":       TOP_K,
            "n_samples":   scores["n_samples"],
            "data_source": source,
            "embed_model": EMBED_MODEL,
        })
        mlflow.log_metrics({
            "faithfulness":     scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
        })
        out = "/tmp/ragas_results.json"
        scores["df"].to_json(out, orient="records", force_ascii=False, indent=2)
        mlflow.log_artifact(out, artifact_path="ragas")
    logger.info(f"✓ MLflow: {run_name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=== RAGAS Evaluation Start ===")

    questions, answers, contexts = load_from_logs()
    source = "real_logs"

    if len(questions) < MIN_REAL_LOGS:
        questions, answers, contexts = build_from_fallback()
        source = "fallback"

    if len(questions) < 3:
        logger.error("Không đủ data. Kiểm tra Qdrant + GROQ_API_KEY.")
        sys.exit(1)

    logger.info(f"Dataset: {len(questions)} samples (source: {source})")
    scores = run_ragas(questions, answers, contexts)
    log_mlflow(scores, source)

    print("\n" + "=" * 45)
    print(f"RAGAS Results [{source}]")
    print("=" * 45)
    print(f"faithfulness:     {scores['faithfulness']:.3f}")
    print(f"answer_relevancy: {scores['answer_relevancy']:.3f}")
    print(f"n_samples:        {scores['n_samples']}")
    print("=" * 45)


if __name__ == "__main__":
    main()
