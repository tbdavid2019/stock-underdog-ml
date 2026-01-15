# Stock Prediction App - Setup Guide

完整的安裝與設定指南。

## 系統需求

- Python 3.11+
- Supabase 帳號
- 8GB+ RAM（用於模型訓練）

## 安裝步驟

### 1. Clone 專案

```bash
git clone https://github.com/tbdavid2019/stock-underdog-ml.git
cd stock-underdog-ml
```

### 2. 建立虛擬環境

```bash
python3.11 -m venv myenv
source myenv/bin/activate
```

### 3. 安裝相依套件

```bash
pip install -r requirements.txt
```

#### 可選套件（Optional）

以下套件為可選功能，**不安裝不影響核心功能**：

**MongoDB 支援**（目前使用 Supabase，不需要）
```bash
pip install pymongo>=4.0.0
```

**MySQL 支援**（目前使用 Supabase，不需要）
```bash
pip install mysql-connector-python>=8.0.0
```

**Chronos 模型（需要 C++ 編譯器 + Python 開發套件）**

> ⚠️ **預設已關閉**：由於安裝複雜，`autogluon.timeseries` 在 `requirements.txt` 中預設已註解。

如果要啟用 Chronos 模型：

```bash
# 1. 安裝系統依賴
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential g++ python3-dev

# CentOS/RHEL/Amazon Linux
sudo yum groupinstall "Development Tools"
sudo yum install -y gcc-c++ python3-devel

# 2. 在 requirements.txt 中取消註解 autogluon.timeseries
# 將 # autogluon.timeseries 改為 autogluon.timeseries

# 3. 安裝套件
pip install autogluon.timeseries

# 4. 在 .env 中啟用
USE_CHRONOS=true
```

> ℹ️ **提示**：如果看到 "MongoDB 功能未啟用" 訊息，可以忽略。這是可選功能，不影響系統運行。


### 4. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env` 填入您的設定：

```bash
# 必填：Supabase
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key

# 選填：通知服務
DISCORD_WEBHOOK_URL=your_webhook_url
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=your_channel_id
```

### 5. 建立 Supabase 資料庫

1. 登入 [Supabase Dashboard](https://app.supabase.com)
2. 建立新專案
3. 到 **SQL Editor**
4. 執行 `supabase_schema.sql` 的內容

### 6. 測試執行

```bash
python main.py
```

## 設定說明

### 模型開關

在 `.env` 中控制要使用的模型：

```bash
USE_PROPHET=true      # Facebook Prophet
USE_CHRONOS=true      # AutoGluon Chronos
USE_CROSS=true        # Cross-sectional models (TabNet, SFM)
USE_TRANSFORMER=false # Transformer (較慢，可選)
```

### 訓練參數

```bash
CROSS_PERIOD=6mo      # 資料下載期間
CROSS_EPOCHS=150      # 訓練 epochs
CHRONOS_PERIOD=6mo    # Chronos 預測期間
```

### 通知設定

#### Discord
1. 建立 Discord Webhook
2. 複製 Webhook URL 到 `.env`

#### Telegram
1. 用 [@BotFather](https://t.me/botfather) 建立 Bot
2. 取得 Bot Token
3. 將 Bot 加入頻道並取得 Channel ID

#### Email
使用 Gmail 的話需要：
1. 啟用 2FA
2. 產生應用程式密碼
3. 填入 `.env`

## 定期執行（Cron）

編輯 crontab：
```bash
crontab -e
```

加入：
```bash
# 每天早上 8 點執行
0 8 * * * /home/ec2-user/stock-underdog-ml/run.sh
```

## 回測功能

### 初次設定
資料庫 schema 已包含回測欄位，無需額外設定。

### 執行回測
```bash
python backtest/backtest.py
```

### 查看結果
```bash
python backtest/analyze_backtest.py
```

或在 Supabase SQL Editor：
```sql
SELECT * FROM model_performance;
```

## 疑難排解

### Keras 版本問題
如果遇到 Keras 3 錯誤：
```bash
pip install tf-keras
```

### Supabase 連線失敗
1. 確認 `SUPABASE_URL` 和 `SUPABASE_SERVICE_KEY` 正確
2. 檢查是否使用 **service_role** key（不是 anon key）
3. 確認資料表已建立

### 記憶體不足
減少並行處理數量，編輯 `main.py`：
```python
max_workers=3  # 從 5 降到 3
```

## 更新專案

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

檢查 `supabase_schema.sql` 是否有更新，如有需要執行新的 migration。

## 支援

- GitHub Issues: https://github.com/tbdavid2019/stock-underdog-ml/issues
- 文檔：查看 `README.md` 和 `backtest/README.md`
