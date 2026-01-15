# Docker 使用說明

## 建立映像

```bash
docker build -t stock-underdog-ml .
```

或使用 docker-compose：

```bash
docker-compose build
```

## 執行

### 方式 1：使用 docker run
```bash main
docker run --rm \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  stock-underdog-ml
```

# 執行 backtest
```
docker run --rm \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  stock-underdog-ml \
  python backtest/backtest.py

```

# 自由bash互動
```
docker run --rm -it \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  stock-underdog-ml bash
```

### 方式 2：使用 docker-compose

```bash
# 前台執行
docker-compose up

# 背景執行
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止
docker-compose down
```

## 定期執行（使用 cron）

在主機上設定 cron job：

```bash
# 編輯 crontab
crontab -e

# 每天早上 8:00 執行
0 8 * * * cd /home/david/stock-underdog-ml && docker-compose run --rm stock-ml python main.py >> logs/cron.log 2>&1
```

## 進入容器除錯

```bash
# 使用 docker-compose
docker-compose run --rm stock-ml bash

# 或使用 docker
docker run --rm -it \
  -v $(pwd)/.env:/app/.env:ro \
  stock-underdog-ml bash
```

## 更新代碼後重新建立

```bash
docker-compose build --no-cache
```

## 注意事項

1. 確保 `.env` 檔案存在並包含所有必要的配置
2. `logs/` 和 `cache/` 目錄會自動建立
3. 使用 Python 3.11 避免 Python 3.13 的相容性問題

## CPU 指令集相容性說明

### 哪些套件需要 AVX 指令集？

**需要 AVX/AVX2 指令集（較新 CPU）：**
- `tensorflow` - 標準版本使用 AVX、AVX2、FMA 等進階指令
- `tensorflow` 2.5+ 版本強制要求 AVX

**不需要 AVX（可在舊 CPU 運行）：**
- `tensorflow-cpu` - CPU 版本，相容性較好
- `torch` (PyTorch) - 一般不強制要求 AVX
- `numpy`, `pandas`, `scikit-learn` - 基礎科學計算套件

### 如何選擇版本？

#### 方案 1：舊 CPU 或虛擬機（無 AVX 支援）
使用 `tensorflow-cpu`（目前 Dockerfile 設定）：

```dockerfile
# Dockerfile 第 14-18 行
RUN pip install --no-cache-dir \
    python-dotenv requests urllib3 yfinance \
    torch torchvision \
    tensorflow-cpu keras tf-keras prophet supabase
```

#### 方案 2：新 CPU 支援 AVX（效能較好）
使用標準 `tensorflow`：

```dockerfile
# 將 tensorflow-cpu 改為 tensorflow
RUN pip install --no-cache-dir \
    python-dotenv requests urllib3 yfinance \
    torch torchvision \
    tensorflow keras tf-keras prophet supabase
```

### 如何檢查 CPU 是否支援 AVX？

```bash
# Linux 檢查
lscpu | grep avx

# 或
cat /proc/cpuinfo | grep avx
```

有輸出 = 支援 AVX，可使用標準 `tensorflow`  
無輸出 = 不支援，必須使用 `tensorflow-cpu`
