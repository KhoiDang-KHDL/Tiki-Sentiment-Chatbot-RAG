ARG CI_BASE_IMAGE=python:3.11-slim-bookworm
FROM ${CI_BASE_IMAGE}
USER root
RUN id -u app >/dev/null 2>&1 || useradd --create-home --uid 10001 app
WORKDIR /opt/tiki
COPY requirements-ci.txt requirements-runtime-ui.txt requirements-runtime-api.txt ./
RUN pip install --no-cache-dir -r requirements-ci.txt && pip check
COPY --chown=app:app runtime/ ./runtime/
RUN mkdir -p .files .chainlit reports/ci && chown -R app:app /opt/tiki
USER app
ENV PYTHONPATH=/opt/tiki PYTHONUNBUFFERED=1 TIKI_LOADTEST_MODE=synthetic API_ENABLE_METRICS=true API_LOG_INTERACTIONS=false HF_HUB_OFFLINE=1 CHAINLIT_NO_TELEMETRY=true QDRANT_COLLECTION=synthetic_fixture
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "runtime.chatbot.loadtest_api:app", "--host", "0.0.0.0", "--port", "8000"]
