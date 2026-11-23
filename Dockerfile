# One image for the scoring API (default), the Streamlit dashboard and the MLflow server.
#   docker build -t credit-risk-platform .
#   docker run -p 8000:8000 credit-risk-platform                       # API -> http://localhost:8000/docs
# The API listens on $PORT when set (Render and similar hosts inject it), else 8000.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    CREDIT_RISK_HOME=/app \
    MODEL_URI=/app/artifacts/model \
    MLFLOW_DISABLE_AGENT_HINT=1

# libgomp1: OpenMP runtime required by LightGBM / XGBoost; curl: health checks
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

WORKDIR /app

# Dependencies first (cached layer), then the project itself
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

COPY api ./api
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /mlflow \
    && chown -R app:app /app /mlflow
USER app

EXPOSE 8000 8501 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
