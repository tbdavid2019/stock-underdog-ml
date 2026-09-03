# 🐳 Docker 現代化部署與容器化指南 (Python 3.12)

本專案提供基於 **Python 3.12 Slim** 之生產級多服務容器化架構，支援即時分析執行、常駐定時排程、Supabase/DuckDB 數據同步與容器內自動化測試。

---

## 🚀 快速上手 (Quick Start)

### 1. 官方多架構預建映像檔 (Docker Hub & GHCR)

GitHub Actions CI/CD 會在每次推送到 `main` 分支時，自動構建支援 **x86_64 (Intel/AMD)** 與 **ARM64 (Apple Silicon M系列 / Raspberry Pi / Ampere)** 的多架構優化映像檔，並同步推播發布至 **Docker Hub** 與 **GitHub Container Registry (GHCR)**：

* **Docker Hub 主位址**：`tbdavid2019/stock-underdog-ml:latest`
* **GHCR 備援位址**：`ghcr.io/tbdavid2019/stock-underdog-ml:latest`

```bash
# 推薦：直接從 Docker Hub 拉取最新多架構映像檔（免本地耗時 15+ 分鐘編譯）
docker pull tbdavid2019/stock-underdog-ml:latest

# 或使用 docker-compose 一鍵拉取所有服務相依映像檔
docker compose pull
```

### 2. 單行指令直接啟動（無需 Clone 原始碼專案）

若只需運行 FastAPI REST API 與 Web 前端操盤看板：

```bash
docker run -d \
  -p 8088:8088 \
  --name stock-ml-api \
  --restart unless-stopped \
  -v $(pwd)/data/storage:/app/data/storage \
  -v $(pwd)/logs:/app/logs \
  tbdavid2019/stock-underdog-ml:latest
```

### 3. 本地客製化構建 (Local Build，僅限二次開發)

若您修改了底層 Python C 延伸模組或 Dockerfile，才需要在本地從原始碼自行構建：

```bash
docker compose build
```

---

## 🛠️ 服務指令總覽

本專案在 `docker-compose.yml` 中配置了 4 大獨立服務：

### 1. 單次執行市場開盤前買進指南 (`stock-ml`)

執行台股 50、台股中型 100 或 美股 S&P 500 之完整量化策略運算，並自動寫入 Supabase、本地 DuckDB 與發送推播：

```bash
# 全市場 (TW + US)
docker compose run --rm stock-ml

# 🇹🇼 台股開盤前指南 (台灣50 + 台灣中型100)
docker compose run --rm stock-ml main --market tw

# 🇺🇸 美股開盤前指南 (S&P 500)
docker compose run --rm stock-ml main --market us
```

### 2. 常駐後台定時排程服務 (`stock-ml-cron`)

啟動容器內建之 Linux Cron 守護進程（依據 `Asia/Taipei` 台北時間自動於台股盤前 08:00 與美股盤前 20:30 定時執行買進指南，無需依賴宿主機 crontab）：

```bash
# 背景啟動排程容器
docker compose up -d stock-ml-cron

# 查看排程日誌
docker compose logs -f stock-ml-cron

# 停止排程容器
docker compose stop stock-ml-cron
```

> **內建排程規則 (`docker/crontab`)**：
> - **08:00 (週一至週五)**：🇹🇼 台股開盤前買進決策指南 (`--market tw`)
> - **20:30 (週一至週五)**：🇺🇸 美股開盤前買進決策指南 (`--market us`)

### 3. Supabase ➔ DuckDB 全量數據導回 (`stock-ml-sync`)

一鍵將 Supabase 雲端現有之全部歷史預測數據分頁同步至本地 DuckDB：

```bash
docker compose run --rm stock-ml-sync
```

### 4. 容器內執行全套單元測試 (`stock-ml-test`)

在乾淨的容器環境內執行 44 項量化單元測試：

```bash
docker compose run --rm stock-ml-test
```

### 5. 進入容器互動偵錯 (Interactive Shell)

```bash
docker compose run --rm stock-ml bash
```

---

## 📂 磁碟掛載與資料持久化

在 `docker-compose.yml` 中已配置以下目錄掛載，確保資料與日誌持久化至宿主機：

| 宿主機路徑 | 容器內路徑 | 說明 |
| :--- | :--- | :--- |
| `./.env` | `/app/.env:ro` | 環境變數配置檔（唯讀掛載） |
| `./data/storage` | `/app/data/storage` | **DuckDB 本地時序庫檔案 (`stock_quant.duckdb`)** |
| `./data/cache` | `/app/data/cache` | 證交所三大法人每日快取 JSON |
| `./cache` | `/app/cache` | 指數成分股與行情 Pickle 快取 |
| `./logs` | `/app/logs` | 應用程式與 Cron 排程日誌 |

---

## ❓ 為什麼選擇 Python 3.12 而非 3.11 或 3.13？

1. **極致效能與穩定性**：Python 3.12 相比 3.11 在直譯器效能、記憶體管理與錯誤回溯上有顯著提升。
2. **機器學習二進位輪子（Wheels）相容性最佳**：
   - 目前 `TensorFlow 2.16+`、`PyTorch 2.3+`、`DuckDB`、`NumPy 2.x`、`Pandas` 皆官方原生支援 Python 3.12。
   - Python 3.13 目前部分 C-extension 與 TensorFlow 官方尚未釋出全量正式 Wheels，因此 **Python 3.12 是當前兼具最新特性與 100% 穩定性的黃金版本**。
