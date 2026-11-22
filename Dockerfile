FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH HF_HOME=/app/data/hf UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev
COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/rag/app.py", "--server.address", "0.0.0.0", "--server.headless", "true"]
