FROM python:3.11.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ui.txt .
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY src/ ./src/

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["chainlit", "run", "src/chatbot/demo_chainlit.py", "--host", "0.0.0.0", "--port", "8001", "--headless"]
