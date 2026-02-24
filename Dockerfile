ARG TORCH_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

FROM ${TORCH_BASE_IMAGE} AS python-builder

WORKDIR /app

# NOTE: Prefer HTTPS apt sources; fallback to mirrors.aliyun.com when apt update fails
# (common on some intranet/transparent-proxy networks).
RUN sed -i \
      's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
      /etc/apt/sources.list \
    && (apt-get -o Acquire::Retries=3 update || ( \
          echo "[apt] update failed; falling back to mirrors.aliyun.com" >&2; \
          sed -i 's|https://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|https://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list; \
          apt-get -o Acquire::Retries=3 update)) \
    && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

# 前端构建
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${TORCH_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN sed -i \
      's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
      /etc/apt/sources.list \
    && (apt-get -o Acquire::Retries=3 update || ( \
          echo "[apt] update failed; falling back to mirrors.aliyun.com" >&2; \
          sed -i 's|https://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|https://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list; \
          apt-get -o Acquire::Retries=3 update)) \
    && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /opt/conda /opt/conda

WORKDIR /app

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/hotwords/ ./data/hotwords/
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN mkdir -p data/models data/uploads data/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
