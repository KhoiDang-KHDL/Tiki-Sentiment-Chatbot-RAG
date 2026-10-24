FROM python:3.11.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# Install torch CPU-only first (lighter), then rest
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "src.chatbot.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
