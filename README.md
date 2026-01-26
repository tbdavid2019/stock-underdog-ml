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
- **玄鐵重劍策略** - 均線技術分析 🆕 基於 MA5/MA60/MA120/MA250

### 雙軌策略系統 🆕
結合深度學習與技術面分析的雙重驗證機制：
- **LSTM 預測** - AI 預測明日股價，自動計算漲幅潛力
- **玄鐵重劍** - MA 均線策略，篩選符合技術面條件的股票
- **雙重符合** - 同時通過兩種策略的高信心標的
- **基本面整合** - 自動抓取 PE/PB/Forward PE/EV/EBITDA 數據


### 關鍵特性
- ✅ **下一日預測** - 預測明天的股價，實用性高
- ✅ **自動回測** - 每日驗證預測準確度
- ✅ **多市場支持** - 台股、美股
- ✅ **並行處理** - 8 個 workers 同時處理 LSTM
- ✅ **技術面篩選** - MA5/MA10/MA60/MA120/MA250 均線策略 🆕
- ✅ **基本面數據** - PE/PB/Forward PE/EV/EBITDA 自動獲取 🆕

- ✅ **Supabase 整合** - 雲端資料庫儲存（含雙軌策略表）🆕
- ✅ **多通知管道** - Discord、Telegram、Email

---

## 🚀 快速開始

### 1. 安裝

```bash
git clone https://github.com/tbdavid2019/stock-underdog-ml.git
cd stock-underdog-ml
# 方法 A：使用自動腳本
bash scripts/setup.sh

# 方法 B：手動設定
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

**新用戶**：直接執行 `supabase_schema.sql`，已包含所有雙軌策略欄位
**舊用戶升級**：執行 `scripts/migration/supabase_add_columns.sql` 添加新欄位

表格結構：
```sql
-- predictions 表格（已包含雙軌策略欄位）
-- 基本欄位：ticker, current_price, predicted_price, potential
-- 策略欄位：strategy_type, model_name
-- 技術面：ma5, ma10, ma60, ma120, ma250, pullback_type
-- 基本面：pe, pb, forward_pe
-- 回測欄位：actual_price, accuracy, percentage_error
```

### 4. 執行

**主程式（雙軌策略）：**
```bash
python main.py
```

**舊版（純 LSTM）：**
```bash
python main_lstm_only.py
```

詳細步驟請參考 **[SETUP.md](SETUP.md)**

---

## 📊 預測邏輯

### LSTM 時間軸

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

### 玄鐵重劍策略 🆕

基於《笑傲江湖》中「重劍無鋒，大巧不工」的理念，使用均線判斷大勢：

**Filter 1 - 大勢向上（MA60 上升趨勢）：**
- MA60 在過去 10 個交易日呈現上升
- 股價位於 MA60 之上

**Filter 2 - 小勢回調（買入時機）：**
- 價格接近或短暫跌破 MA60 (±5% 容忍)
- 或接近 MA120 (±5% 容忍)

**回調類型判讀：**

| 回調類型 | 含義 | 買點強度 | 說明 |
|---------|------|---------|------|
| **MA60回調** | 價格在 MA60 附近 ±5% | ⭐⭐⭐ 較激進 | 剛從中期均線反彈，適合積極型投資者 |
| **MA120回調** | 價格在 MA120 附近 ±5% | ⭐⭐⭐⭐ 較穩健 | 價格回到更強支撐位，風險相對較低 |

**技術指標計算：**
- MA5：5 日均線（短期趨勢）
- MA10：10 日均線（短期趨勢確認）
- MA60：60 日均線（中期趨勢，大勢判斷核心）
- MA120：120 日均線（中期支撐）
- MA250：250 日均線（長期價值中樞）

**參數設定：**
```python
lookback = 10      # MA60 斜率檢查天數
tolerance = 0.05   # ±5% 回調容忍範圍
```

**預期通過率：** 約 15-20%（嚴格篩選）

### 雙軌策略整合 🆕

同時執行兩種分析方法，輸出三種結果：

1. **玄鐵重劍** - 僅通過 MA 均線篩選
2. **LSTM預測** - 僅 AI 預測有潛力
3. **雙重符合** - 同時符合兩種策略（最高信心）

### 預測內容

- **輸入**：過去 90 個交易日的 OHLCV 資料
- **輸出**：下一個交易日的收盤價
- **潛力計算**：`(預測價 - 當前價) / 當前價 × 100%`
- **基本面數據** 🆕：PE (本益比), PB (股價淨值比), Forward PE (預估本益比), EV/EBITDA (企業價值倍數)


### 回測驗證

- **頻率**：每日執行（建議用 cron）
- **方法**：比對預測價 vs 實際收盤價
- **指標**：方向準確度、平均誤差、誤差分布

詳細說明請參考 **[backtest/README.md](backtest/README.md)**

### 📊 基本面指標更新頻率 🆕

本系統使用三個關鍵基本面指標：**PE (本益比)**、**PB (股價淨值比)**、**EV/EBITDA (企業價值倍數)**

#### 更新機制說明

所有三個指標都是**比率**，其數值每天變動，但使用的是**最近一季的財報數據**：

| 指標 | 公式 | 更新頻率 | 說明 |
|------|------|----------|------|
| **PE** | 股價 ÷ 每股盈餘 (EPS) | 股價每秒更新<br>EPS 每季更新 | 買這家公司的股票，要付幾倍的年盈餘 |
| **PB** | 股價 ÷ 每股淨值 | 股價每秒更新<br>淨值每季更新 | 買這家公司的股票，要付幾倍的淨資產 |
| **EV/EBITDA** | 企業價值 ÷ EBITDA | 企業價值每天更新<br>EBITDA 每季更新 | 買下整家公司，要付幾倍的營運獲利 |

#### 為什麼每天執行有意義？

雖然財報數據每季才更新一次，但**股價每天變動**，因此：

1. **捕捉價格變化**：股價下跌 → PE/PB/EV 變低 → 相對便宜 ✅
2. **識別買點機會**：「股價回調但基本面穩定」的情況
3. **穩定的比較基準**：財報數據 3 個月才更新，提供穩定的評估基礎

#### 實際範例

以台積電 (2330.TW) 為例：

```
2025-12-31: Q4 財報公布 📊
  └─ EPS = $66.18, EBITDA = $2.6B, 淨值 = $200

2026-01-01: 
  └─ PE = 股價($1800) ÷ EPS($66.18) = 27.2
  └─ EV/EBITDA = 企業價值 ÷ EBITDA($2.6B) = 16.6

2026-01-10: 股價下跌
  └─ PE = 股價($1750) ÷ EPS($66.18) = 26.4  ← 變便宜！
  └─ EV/EBITDA = 16.2  ← 也變便宜！

2026-03-31: Q1 財報公布 📊
  └─ EPS = $70.00 (更新)
  
2026-04-01: 
  └─ PE = 股價($1800) ÷ EPS($70.00) = 25.7  ← 新基準
```

#### EV/EBITDA 的優勢

相比 PE/PB，EV/EBITDA 有以下優點：

- ✅ **考慮債務結構**：企業價值 = 市值 + 負債 - 現金
- ✅ **不受會計政策影響**：EBITDA 排除折舊和攤銷
- ✅ **適合跨產業比較**：尤其是資本密集型產業（製造業、航空業）
- ✅ **更全面的估值**：同時考慮股權和債務

#### 估值標準參考

| 指標 | 便宜 | 合理 | 稍貴 | 昂貴 |
|------|------|------|------|------|
| **PE** | < 15 | 15-20 | 20-25 | > 25 |
| **PB** | < 1.5 | 1.5-3 | 3-5 | > 5 |
| **EV/EBITDA** | < 10 | 10-15 | 15-20 | > 20 |

**注意**：不同產業標準不同，科技股通常估值較高

#### 數據來源

所有基本面數據來自 **Yahoo Finance**，透過 `yfinance` 套件自動獲取，無需手動更新。

台灣50/台灣中型100 成分股與名稱來自 `https://answerbook.david888.com`，通知與報表會顯示「代碼 + 公司名稱」。

---


## 🎯 雙軌策略詳解 🆕

### 設計理念

**為何需要雙軌策略？**
- LSTM 善於捕捉價格變化模式，但可能忽略技術面訊號
- 均線策略基於成熟的技術分析，但缺乏 AI 的預測能力
- 結合兩者可提供**雙重驗證**，降低誤判風險

### 執行流程

```
1. 下載股票列表 (台灣50/SP500...)
            │
            ├─────────────────────┬─────────────────────┐
            ▼                     ▼                     ▼
    2a. LSTM 預測          2b. 玄鐵篩選          3. 計算重疊
    (並行 8 workers)       (MA 均線檢查)
    - 訓練模型               - MA60 上升趨勢
    - 預測明日價             - MA60/MA120 回調
    - 計算潛力 %             - 獲取 MA5/MA10/MA250
    - 抓取 PE/PB             - 抓取 PE/PB
            │                     │                     │
            └─────────────────────┴─────────────────────┘
                                  ▼
            4. 三種輸出結果
            ┌─────────────────────────────────────┐
            │ 🎯 玄鐵重劍 (技術面符合)           │
            │ 🧠 LSTM預測 (AI 看好)              │
            │ ⭐ 雙重符合 (兩者皆符合，最高信心) │
            └─────────────────────────────────────┘
                                  ▼
            5. 保存到 Supabase + 發送通知
```

### 策略參數

**玄鐵重劍策略：**
```python
lookback = 10      # MA60 上升趨勢檢查天數
tolerance = 0.05   # ±5% 回調容忍範圍
period = "1y"      # 使用 1 年數據計算均線
```

**LSTM 策略：**
```python
sequence_length = 90  # 使用 90 天數據訓練
epochs = 20           # 訓練輪數
period = "6mo"        # 使用 6 個月數據
```

### 輸出格式

執行 `python main.py` 後會看到：

```
========================================
📊 台灣50 雙軌策略報告
========================================

🎯 玄鐵重劍策略 (3 檔)
股票       現價      MA60      MA120     回調類型    PE    PB
─────────────────────────────────────────────────────────────
2330.TW    1670.00   1650.00   1600.00   MA60回調   24.5  8.2
2317.TW     234.50    230.00    225.00   MA60回調   18.3  3.1
2454.TW    1234.50   1220.00   1200.00   MA120回調  15.8  2.9

🧠 LSTM 預測策略 (8 檔)
股票       現價      預測價     潛力      PE    Forward PE
────────────────────────────────────────────────────────────
2454.TW    1234.50   1280.00   3.68%    15.2    14.8
2412.TW     456.00    478.00   4.82%    22.1    20.5
...

⭐ 雙重符合 (2 檔) ← 最高信心
股票       現價      預測價    潛力     MA60      PE    PB
─────────────────────────────────────────────────────────────
2330.TW    1670.00   1720.00   3.00%   1650.00  24.5  8.2
2454.TW    1234.50   1280.00   3.68%   1220.00  15.8  2.9

執行時間: 45.2 秒
```

### 如何解讀

1. **只看雙重符合**：保守型投資者，只關注同時通過兩種策略的股票
2. **玄鐵 + LSTM**：參考兩者交集，手動篩選
3. **單看玄鐵**：偏好技術面分析
4. **單看 LSTM**：信任 AI 預測

### 配置通知

在 `.env` 中設定：
```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Discord
DISCORD_WEBHOOK_URL=your_webhook_url

# Email (SMTP)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_TO=recipient@example.com
```

### Supabase 設定

1. 進入 [Supabase Dashboard](https://app.supabase.com)
2. 選擇專案 → SQL Editor
3. 執行 `supabase_dual_strategy_schema.sql`
4. 確認表格建立：`dual_strategy_predictions`

表格欄位：
```sql
- index_name          指數名稱
- strategy_type       策略類型 (玄鐵重劍/LSTM預測/雙重符合)
- ticker              股票代碼
- current_price       當前價格
- predicted_price     LSTM 預測價格
- potential           預測漲幅 %
- ma5, ma10, ma60,    各均線值
  ma120, ma250
- pullback_type       回調類型 (MA60回調/MA120回調)
- pe, pb, forward_pe  基本面數據
- period              數據週期
- timestamp           分析時間
```

---

## 📁 專案結構

```
stock-underdog-ml/
├── README.md              ← 你在這裡
├── SETUP.md              ← 安裝指南
├── .env.example          ← 環境變數範本
│
├── main.py               ← 主程式（雙軌策略）🆕
├── main_lstm_only.py     ← 單一 LSTM 模型（舊版保留）
│
├── config.py             ← 設定管理
├── database.py           ← Supabase 整合（寫入 predictions 表格）🆕
├── parallel_processor.py ← 並行處理
├── data_loader.py        ← 數據下載（含 12hr 快取）🆕
│
├── xuantie_strategy.py   ← 玄鐵重劍均線策略 🆕
├── notifier_dual.py      ← 雙軌策略通知器 🆕
│
├── models/               ← 預測模型
│   └── lstm.py          ← LSTM 下一日預測
│
├── cache/               ← 資料快取 🆕
│   └── stock_data/      ← 股票數據 pickle 快取（12小時）
│
├── output/              ← 輸出文件 🆕
│   └── xuantie_signals_*.csv  ← 玄鐵策略篩選結果
│
├── backtest/            ← 回測系統
│   ├── README.md        ← 回測說明
│   ├── backtest.py      ← 回測腳本
│   └── run_backtest.sh  ← 自動化腳本
│
├── scripts/             ← 測試和維護腳本
│   ├── migration/       ← 數據庫遷移腳本（舊版升級用）🆕
│   ├── verification/    ← 驗證腳本 🆕
│   ├── README.md
│   └── [其他測試腳本...]
│
├── supabase_schema.sql  ← 完整資料庫結構（已包含雙軌策略欄位）🆕
│
└── logs/                ← 執行日誌
```

---

## 📋 檔案說明

### 主程式
- **main.py** - 雙軌策略主程式（LSTM + 玄鐵重劍）
- **main_lstm_only.py** - 純 LSTM 版本（向後兼容）

### 策略模組
- **xuantie_strategy.py** - MA 均線策略（MA5/MA10/MA60/MA120/MA250）
- **notifier_dual.py** - 多平台通知（Telegram/Discord/Email）

### 資料庫
- **supabase_schema.sql** - 完整資料庫結構
  - 包含：predictions 表格（已擴展）
  - 欄位：strategy_type, ma5-ma250, pe/pb/forward_pe
  - 索引：針對雙軌策略查詢優化
- **database.py** - Python 資料庫操作
  - 寫入 predictions 表格
  - 支援三種策略類型（玄鐵重劍/LSTM預測/雙重符合）

### 遷移腳本（舊版用戶升級）
- **scripts/migration/supabase_add_columns.sql** - 為現有表格添加新欄位
- **scripts/verification/verify_predictions_extended.py** - 驗證表格結構

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

**單一 LSTM 模型：**
```sql
SELECT ticker, predicted_price, potential 
FROM predictions 
WHERE timestamp > NOW() - interval '1 day'
ORDER BY potential DESC
LIMIT 10;
```

**雙軌策略** 🆕：
```sql
-- 查詢雙重符合（最高信心）
SELECT ticker, strategy_type, current_price, predicted_price, potential,
       ma5, ma60, ma120, ma250, pullback_type, pe, pb
FROM dual_strategy_predictions 
WHERE strategy_type = '雙重符合'
  AND timestamp > NOW() - interval '1 day'
ORDER BY potential DESC;

-- 查詢各策略統計
SELECT strategy_type, COUNT(*) as count
FROM dual_strategy_predictions
WHERE timestamp > NOW() - interval '1 day'
GROUP BY strategy_type;
```

### 通知（Discord/Telegram）

**單一模型：**
```
🥇 前五名 LSTM 🧠
股票     現價      預測價     潛力
----------------------------------------
2330.TW  1670.00   1720.00    3.00%
2317.TW   234.50    245.00    4.48%
...
```

**雙軌策略** 🆕：
```
========================================
📊 台灣50 雙軌策略報告
========================================

🎯 玄鐵重劍策略 (MA均線篩選)
股票      現價      MA60      回調類型    PE    PB
────────────────────────────────────────────
2330.TW   1670.00   1650.00   MA60回調   24.5  8.2
2317.TW    234.50    230.00   MA60回調   18.3  3.1

🧠 LSTM 預測策略
股票      現價      預測價     潛力      PE    Forward PE
─────────────────────────────────────────────────────
2454.TW   1234.50   1280.00   3.68%    15.2    14.8
2412.TW    456.00    478.00   4.82%    22.1    20.5

⭐ 雙重符合 (高信心)
股票      現價      預測價    潛力     MA60      PE    PB
───────────────────────────────────────────────────
2330.TW   1670.00   1720.00   3.00%   1650.00  24.5  8.2

執行時間: 45.2 秒
```

---

## 🔄 自動化執行

### Cron 設定

**單一 LSTM 模型：**
```bash
crontab -e

# 每天早上 8 點執行預測
0 8 * * * /home/ec2-user/stock-underdog-ml/run.sh

# 每天早上 9 點執行回測（約 3 分鐘）
0 9 * * * /home/ec2-user/stock-underdog-ml/backtest/run_backtest.sh
```

**雙軌策略** 🆕：
```bash
crontab -e

# 每天台股收盤後 15:00 執行雙軌分析
0 15 * * 1-5 cd /home/ec2-user/stock-underdog-ml && source myenv/bin/activate && python main_dual_strategy.py >> logs/dual_strategy.log 2>&1

# 或使用 bash 腳本
0 15 * * 1-5 /home/ec2-user/stock-underdog-ml/run_dual.sh
```

**run_dual.sh 範例：**
```bash
#!/bin/bash
cd /home/ec2-user/stock-underdog-ml
source myenv/bin/activate
python main_dual_strategy.py
deactivate
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
   - 單一模型：執行 `supabase_schema.sql`
   - 雙軌策略：執行 `supabase_dual_strategy_schema.sql` 🆕

### 記憶體不足
編輯 `main_dual_strategy.py` 降低並行數：
```python
max_workers=4  # 從 8 降到 4
```

### 玄鐵策略通過率 0% 🆕
可能原因：
- 市場處於下跌趨勢（MA60 未上升）
- 參數太嚴格

調整參數：
```python
# 在 main_dual_strategy.py 中
xuantie_stocks = filter_stocks_by_xuantie(
    stock_list,
    period="1y",
    lookback=7,      # 降低至 7 天
    tolerance=0.08   # 放寬至 ±8%
)
```

### 數據快取問題 🆕
快取位置：`cache/stock_data/`
快取有效期：12 小時

清除快取：
```bash
rm -rf cache/stock_data/*
```

更多問題請參考 **[SETUP.md](SETUP.md)** 的疑難排解章節

---

## 📝 更新日誌

### v2.1 (2025-01-22) 🆕
- ✨ **新增雙軌策略系統**：結合 LSTM + 玄鐵重劍 MA 均線策略
- ✨ **玄鐵重劍策略**：基於 MA5/MA10/MA60/MA120/MA250 的技術面篩選
  - Filter 1：MA60 上升趨勢（10 天）+ 股價在 MA60 之上
  - Filter 2：價格回調至 MA60/MA120 (±5%)
  - 通過率：約 15-20%
- ✨ **基本面整合**：自動抓取 PE/PB/Forward PE 數據
- ✨ **數據快取系統**：12 小時 pickle 快取，加速重複查詢
- ✨ **雙軌策略資料表**：包含 ma5, ma10, ma60, ma120, ma250, pe, pb, forward_pe 欄位
- ✨ **多平台通知**：Telegram (HTML)、Discord (Markdown)、Email
- 🔧 優化 LSTM 並行處理：8 workers 同時運行
- 📊 新增 `main_dual_strategy.py` 雙軌策略入口
- 📊 新增 `xuantie_strategy.py` 均線策略模組
- 📊 新增 `notifier_dual.py` 雙軌通知器
- 📊 新增 `data_loader.py` 快取系統

### v2.0 (2026-01-08)
- ✨ **重大改進**：改為預測下一交易日（原為歷史最大值）
- ✨ 新增自動回測系統
- ✨ 優化 LSTM 架構（3 層，128→64→32 units）
- ✨ 增加訓練 epochs（10→100，後優化為 20）
- 🐛 修復資料洩漏問題
- 🔧 關閉 Prophet 和 Chronos（準確度不佳）
- 📁 整理專案結構（新增 scripts/ 資料夾）

### v1.0
- 初始版本

---

## �️ 檔案結構整理

### 目錄分類

**根目錄** - 主程式和核心模組
- `main.py` - 主程式（雙軌策略）
- `main_lstm_only.py` - 舊版保留

**cache/** - 數據快取（自動生成）
- `stock_data/` - 股票數據 pickle 快取（12小時）

**output/** - 輸出文件（自動生成）
- `xuantie_signals_*.csv` - 玄鐵策略篩選結果

**logs/** - 執行日誌（自動生成）
- `app.log` - 主程式日誌

**scripts/** - 維護和測試腳本
- `migration/` - 數據庫遷移（舊版升級用）
- `verification/` - 表格驗證腳本
- 其他測試腳本

**backtest/** - 回測系統
- `backtest.py` - 回測腳本
- `run_backtest.sh` - 自動化腳本

**models/** - AI 模型
- `lstm.py` - LSTM 模型

### 數據庫檔案

**新用戶（推薦）：**
- `supabase_schema.sql` - 執行此檔案，已包含所有欄位

**舊用戶升級：**
- `scripts/migration/supabase_add_columns.sql` - 添加雙軌策略欄位
- `scripts/verification/verify_predictions_extended.py` - 驗證表格結構

### 重要提示

1. **單一資料表** - 所有數據（LSTM + 玄鐵策略）寫入同一張 `predictions` 表格
2. **向後兼容** - 舊數據 `strategy_type = NULL`，新數據有明確策略類型
3. **回測兼容** - 因為使用同一張表，回測功能完全兼容
4. **自動生成** - `cache/`, `output/`, `logs/` 會自動創建，不需手動建立

---
## ⏰ 自動化與排程

### 環境說明

本專案使用 **conda stockml** 環境，請確保排程腳本正確啟動環境。

### 快速設置排程

#### 方法 1：使用互動式設置腳本（推薦）

```bash
./setup_cron.sh
```

腳本會詢問您想要的執行時間：
- **選項 1**：台股收盤後 14:30（週一至週五）
- **選項 2**：美股收盤後 06:30（週一至週五）
- **選項 3**：每日午夜 00:00
- **選項 4**：自訂時間

#### 方法 2：手動設置 crontab

```bash
crontab -e
```

添加以下其中一行：

```bash
# 台股收盤後執行（週一至週五 14:30）
30 14 * * 1-5 /path/to/stock-underdog-ml/run_daily.sh >> /path/to/stock-underdog-ml/logs/cron.log 2>&1

# 美股收盤後執行（週一至週五 6:30）
30 6 * * 1-5 /path/to/stock-underdog-ml/run_daily.sh >> /path/to/stock-underdog-ml/logs/cron.log 2>&1

# 每日午夜執行（使用快取數據）
0 0 * * * /path/to/stock-underdog-ml/run_daily.sh >> /path/to/stock-underdog-ml/logs/cron.log 2>&1
```

### 執行腳本說明

#### run_daily.sh（每日自動執行）

**功能：**
1. ✅ 自動啟動 stockml conda 環境
2. ✅ 執行回測（驗證昨日預測）
3. ✅ 執行雙軌策略分析（今日預測）
4. ✅ 清理 30 天前的舊日誌
5. ✅ 所有輸出記錄到 logs/daily_YYYYMMDD_HHMMSS.log

**手動測試：**
```bash
./run_daily.sh
```

**查看執行日誌：**
```bash
# 即時查看最新日誌
tail -f logs/daily_*.log

# 查看 cron 執行記錄
tail -f logs/cron.log

# 列出最近 10 次執行
ls -lt logs/daily_*.log | head -10
```

### 環境變數檢查

確保 `.env` 檔案已正確設置：

```bash
# 必填項目
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key

# 通知（選填）
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHANNEL_ID=your_channel_id
DISCORD_WEBHOOK_URL=your_webhook
SENDER_EMAIL=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
TO_EMAILS=recipient1@example.com,recipient2@example.com
```

### 排程執行流程

```
Cron 觸發 (例如：14:30)
         ↓
啟動 run_daily.sh
         ↓
載入 conda stockml 環境
         ↓
┌────────┴────────┬───────────────┐
│                 │               │
執行回測      執行雙軌策略    清理舊日誌
(驗證昨日)    (預測今日)    (保留30天)
         ↓
記錄到 logs/daily_YYYYMMDD_HHMMSS.log
         ↓
發送通知 (Telegram/Discord/Email)
         ↓
完成
```

### 日誌管理

**日誌類型：**
- `logs/daily_*.log` - 每日執行主日誌（自動生成時間戳記）
- `logs/cron.log` - Cron 執行記錄（包含錯誤訊息）
- `logs/app.log` - 應用程式運行日誌

**自動清理：**
- 超過 30 天的日誌會自動刪除
- 保持 logs/ 目錄整潔

### 常見問題

#### Q1: Cron 執行失敗，找不到 conda？

**A:** 確保 `run_daily.sh` 中 conda 路徑正確：
```bash
# 檢查您的 conda 安裝路徑
which conda

# 修改 run_daily.sh 第 7 行
source /path/to/your/miniconda3/etc/profile.d/conda.sh
```

#### Q2: 環境沒有正確啟動？

**A:** 手動測試環境啟動：
```bash
source /path/to/your/miniconda3/etc/profile.d/conda.sh
conda activate stockml
python --version  # 確認 Python 版本
pip list | grep tensorflow  # 確認套件安裝
```

#### Q3: 如何確認 Cron 設置成功？

**A:** 檢查 crontab 列表：
```bash
crontab -l
```

#### Q4: 如何暫停自動執行？

**A:** 註解掉 crontab 中的行：
```bash
crontab -e
# 在行首加上 # 註解
# 30 14 * * 1-5 /path/to/stock-underdog-ml/run_daily.sh ...
```

#### Q5: 執行時間建議？

**A:** 建議時間：
- **台股為主**：14:30（收盤後 30 分鐘，數據已更新）
- **美股為主**：06:30（美東時間收盤後，台灣早上）
- **兩者都做**：設置兩個排程分別執行

### 監控與維護

**檢查系統狀態：**
```bash
# 查看 Cron 服務狀態
systemctl status cron

# 查看最近執行結果
tail -100 logs/cron.log | grep -E "(✅|❌|完成|失敗)"

# 檢查資料庫連線
python -c "from database import SupabaseManager; db = SupabaseManager(); print('✅ 連線成功' if db.enabled else '❌ 連線失敗')"
```

**效能監控：**
```bash
# 查看執行時間
grep "執行時間\|結束時間" logs/daily_*.log | tail -20

# 查看預測數量
grep "符合條件\|預測完成\|雙重符合" logs/daily_*.log | tail -10
```

---

## 📊 結果解讀指南

完整的輸出解讀邏輯請參考 **[explain.md](explain.md)**，包含：
- 回調類型判讀（MA60 vs MA120）
- PE/PB 估值分析
- LSTM 預測強度評級
- 綜合決策矩陣
- 風險等級分類（S/A/B/C/D）

---
## �📄 授權

MIT License

## 👤 作者

David (tbdavid2019)

## 🙏 致謝

- Yahoo Finance - 股價資料
- Supabase - 資料庫服務
- TensorFlow - LSTM 框架
