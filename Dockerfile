# RankLens API — FastAPI service (api/main.py) on port 4711.
# Single-stage: all deps ship manylinux wheels (numpy/scipy/lxml/curl_cffi), so
# no compiler toolchain is needed. curl is added for the container healthcheck.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl for the healthcheck; clean apt lists to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# App source.
COPY ranklens ./ranklens
COPY api ./api

# SQLite store + rendered reports live here — mount a volume to persist across deploys.
RUN mkdir -p /app/data
EXPOSE 4711

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:4711/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "4711"]
