FROM python:3.11-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /opt/tiki
COPY requirements-runtime-ui.txt /tmp/requirements-runtime-ui.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime-ui.txt && pip check
COPY --chown=app:app runtime/chatbot/demo_chainlit.py runtime/chatbot/demo_chainlit_http.py runtime/chatbot/api_client.py runtime/chatbot/container_check.py runtime/chatbot/monitoring_check.py ./runtime/chatbot/
COPY --chown=app:app chainlit.md ./chainlit.md
COPY --chown=app:app .chainlit/config.toml ./.chainlit/config.toml
RUN mkdir -p .files && chown -R app:app /opt/tiki
USER app
ENV PYTHONPATH=/opt/tiki PYTHONUNBUFFERED=1 CHAINLIT_NO_TELEMETRY=true
EXPOSE 8001
CMD ["python", "-m", "chainlit", "run", "runtime/chatbot/demo_chainlit.py", "--host", "0.0.0.0", "--port", "8001", "--headless"]
