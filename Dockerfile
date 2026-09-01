# ==============================================================================
# Stock Prediction & Multi-Strategy Quantitative Platform
# Production-Grade Multi-Service Dockerfile (Python 3.12 Slim)
# ==============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Taipei \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. 安裝系統基礎工具 (時區、GCC 編譯、Cron 排程器、Git、Curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    tzdata \
    cron \
    ca-certificates \
    && ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# 2. 安裝 Python 依賴
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# 3. 複製應用程式原始碼
COPY . .

# 4. 建立持久化目錄與腳本權限
RUN mkdir -p logs cache data/storage data/cache models && \
    chmod +x docker/entrypoint.sh 2>/dev/null || true

# 5. 設定 Entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# 預設執行主程序日報
CMD ["main"]
