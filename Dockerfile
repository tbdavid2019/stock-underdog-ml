# ==========================================
# Stock Prediction & Multi-Strategy ML Pipeline
# Modern Production Dockerfile (Python 3.11 Slim)
# ==========================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Taipei \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 安裝系統必要編譯、網路與時區工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製應用程式原始碼
COPY . .

# 建立日誌、快取與 DuckDB 本地儲存目錄
RUN mkdir -p logs cache data/storage data/cache models

# 預設啟動命令
CMD ["python", "main.py"]
