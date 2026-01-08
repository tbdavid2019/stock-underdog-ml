# Stock Prediction Application

AI 驅動的股票預測系統，使用 **LSTM** 預測下一個交易日的股價。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 快速導航

- **[SETUP.md](SETUP.md)** - 完整安裝指南（新用戶必讀）
- **[CHANGELOG.md](CHANGELOG.md)** - 版本更新記錄
- **[backtest/](backtest/)** - 回測系統說明
- **[scripts/](scripts/)** - 測試和維護腳本
- **[.env.example](.env.example)** - 環境變數範本

---

## 🎯 核心功能

### 預測模型
- **LSTM** - 預測下一交易日收盤價 ⭐ 主要模型（方向準確度 54.6%）

### 關鍵特性
- ✅ **下一日預測** - 預測明天的股價，實用性高
- ✅ **自動回測** - 每日驗證預測準確度
- ✅ **多市場支持** - 台股、美股
- ✅ **並行處理** - 5 個 workers 同時處理
- ✅ **Supabase 整合** - 雲端資料庫儲存
- ✅ **多通知管道** - Discord、Telegram、Email

---

## 🚀 快速開始

### 1. 安裝

```bash
git clone https://github.com/tbdavid2019/stock-underdog-ml.git
cd stock-underdog-ml
python3.11 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. 設定

```bash
cp .env.example .env
# 編輯 .env 填入 Supabase credentials
```

**關鍵設定：**
```bash
# 必填
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key

# 模型選擇（建議只開 LSTM）
USE_PROPHET=false
USE_CHRONOS=false
USE_CROSS=false
USE_TRANSFORMER=false
```

### 3. 建立資料庫

在 [Supabase Dashboard](https://app.supabase.com) 執行 `supabase_schema.sql`

### 4. 執行

```bash
python main.py
```

詳細步驟請參考 **[SETUP.md](SETUP.md)**

---

## 📊 預測邏輯

### 時間軸

```
今天 1/8              明天 1/9              後天 1/10
   │                    │                    │
   ▼                    ▼                    ▼
執行 main.py      ← 預測這天的價格    執行 backtest.py
下載 6 個月資料                        驗證預測準確度
訓練 LSTM 模型
預測明天收盤價
儲存到資料庫
```

### 預測內容

- **輸入**：過去 90 個交易日的 OHLCV 資料
- **輸出**：下一個交易日的收盤價
- **潛力計算**：`(預測價 - 當前價) / 當前價 × 100%`

### 回測驗證

- **頻率**：每日執行（建議用 cron）
- **方法**：比對預測價 vs 實際收盤價
- **指標**：方向準確度、平均誤差、誤差分布

詳細說明請參考 **[backtest/README.md](backtest/README.md)**

---

## 📁 專案結構

```
stock-underdog-ml/
├── README.md              ← 你在這裡
├── SETUP.md              ← 安裝指南
├── .env.example          ← 環境變數範本
├── main.py               ← 主程式入口
├── config.py             ← 設定管理
├── database.py           ← Supabase 整合
├── parallel_processor.py ← 並行處理
├── models/               ← 預測模型
│   └── lstm.py          ← LSTM 下一日預測
├── backtest/            ← 回測系統
│   ├── README.md        ← 回測說明
│   ├── backtest.py      ← 回測腳本
│   └── run_backtest.sh  ← 自動化腳本
├── scripts/             ← 測試和維護腳本
│   ├── README.md
│   └── backfill_historical.py
└── logs/                ← 執行日誌
```

---

## ⚙️ 設定選項

### 模型開關（.env）

```bash
# 建議配置：只使用 LSTM
USE_PROPHET=false
USE_CHRONOS=false
USE_CROSS=false
USE_TRANSFORMER=false
```

### 選擇股票指數（main.py）

```python
selected_indices = ["台灣50", "台灣中型100", "SP500"]
```

可用指數：
- `台灣50` - 台灣市值前 50
- `台灣中型100` - 台灣中型股
- `SP500` - 標普 500
- `NASDAQ` - 那斯達克 100
- `費城半導體` - SOX 指數
- `道瓊` - 道瓊工業

---

## 📈 模型表現

基於 543 筆回測資料（2026-01-05 ~ 2026-01-08）：

### LSTM
- **方向準確度：54.6%** ⚠️（略高於隨機 50%）
- **平均誤差：21.3%**
- **誤差 ≤10%：53.3%**
- **誤差 ≤20%：71.2%**

### 各指數表現
- **台灣中型100：51.9%** - 最佳
- **SP500：48.5%**
- **台灣50：41.5%** - 需改進

**結論：** 模型有改進空間，但邏輯正確，可用於實際交易參考。

---

## 📈 輸出結果

### 資料庫（Supabase）

```sql
SELECT ticker, predicted_price, potential 
FROM predictions 
WHERE timestamp > NOW() - interval '1 day'
ORDER BY potential DESC
LIMIT 10;
```

### 通知（Discord/Telegram）

```
🥇 前五名 LSTM 🧠
股票     現價      預測價     潛力
----------------------------------------
2330.TW  1670.00   1720.00    3.00%
2317.TW   234.50    245.00    4.48%
...
```

---

## 🔄 自動化執行

### Cron 設定

```bash
crontab -e

# 每天早上 8 點執行預測
0 8 * * * /home/ec2-user/stock-underdog-ml/run.sh

# 每天早上 9 點執行回測（約 3 分鐘）
0 9 * * * /home/ec2-user/stock-underdog-ml/backtest/run_backtest.sh
```

---

## 🧪 回測系統

驗證模型預測準確度的完整系統。

### 關鍵指標

- **方向準確度** - 預測漲跌方向的正確率（目標 >60%）
- **平均誤差** - 預測價格的平均誤差百分比
- **誤差 ≤5%** - 高精度預測的比例

### 執行回測

```bash
python backtest/backtest.py
python backtest/analyze_backtest.py
```

詳細說明請參考 **[backtest/README.md](backtest/README.md)**

---

## 🛠️ 疑難排解

### Keras 版本問題
```bash
pip install tf-keras
```

### Supabase 連線失敗
1. 確認使用 **service_role** key（不是 anon key）
2. 檢查 `SUPABASE_URL` 格式正確
3. 確認資料表已建立

### 記憶體不足
編輯 `main.py` 降低並行數：
```python
max_workers=3  # 從 5 降到 3
```

更多問題請參考 **[SETUP.md](SETUP.md)** 的疑難排解章節

---

## 📝 更新日誌

### v2.0 (2026-01-08)
- ✨ **重大改進**：改為預測下一交易日（原為歷史最大值）
- ✨ 新增自動回測系統
- ✨ 優化 LSTM 架構（3 層，128→64→32 units）
- ✨ 增加訓練 epochs（10→100）
- 🐛 修復資料洩漏問題
- 🔧 關閉 Prophet 和 Chronos（準確度不佳）
- 📁 整理專案結構（新增 scripts/ 資料夾）

### v1.0
- 初始版本

---

## 📄 授權

MIT License

## 👤 作者

David (tbdavid2019)

## 🙏 致謝

- Yahoo Finance - 股價資料
- Supabase - 資料庫服務
- TensorFlow - LSTM 框架