FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements-runtime-api.txt requirements-runtime-ml.txt ./
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-runtime-ml.txt -r requirements-runtime-api.txt \
    && pip check
COPY --chown=app:app runtime/ ./runtime/
COPY --chown=app:app scripts/update_kb.py ./scripts/update_kb.py
RUN mkdir -p logs && chown app:app logs
USER app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "runtime.chatbot.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
