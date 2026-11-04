# 回測系統

驗證**下一交易日預測**準確度的自動化系統。

📖 **返回主文檔**: [../README.md](../README.md)

---

## 📊 回測原理

### 預測與驗證流程

```
1/8 (三) 執行 main.py
   ↓
   預測 1/9 (四) 的收盤價
   記錄：current_price=100, predicted_price=105
   ↓
1/9 (四) 市場收盤
   ↓
1/9 (四) 晚上執行 backtest.py
   ↓
   抓取 1/9 實際收盤價 = 103
   計算誤差：(105-103)/103 = 1.94%
   更新資料庫
```

### 自動處理假日

系統使用 yfinance 自動判斷下一個交易日：
- ✅ 週末自動跳過
- ✅ 國定假日自動跳過
- ✅ 台股/美股假日自動區分

**範例：**
```
1/10 (五) 預測 → 下一交易日 = 1/13 (一)
1/19 (日，春節) 預測 → 下一交易日 = 1/29 (三)
```

---

## 📁 檔案說明

### SQL Schema
- `supabase_migration_safe.sql` - 安全的 schema 升級（已有資料庫時用）
- `supabase_schema_backtest.sql` - 完整 schema（舊版，已整合到主 schema）

### Python Scripts
- `backtest.py` - **主要回測腳本**，下載實際價格並計算準確度
- `analyze_backtest.py` - 分析回測結果，產生模型表現報告
- `run_backtest.sh` - 自動化執行腳本（用於 cron）

---

## 🚀 使用方式

### 手動執行

```bash
cd /home/ec2-user/stock-underdog-ml
source myenv/bin/activate
python backtest/backtest.py
```

### 自動化（推薦）

```bash
# 編輯 crontab
crontab -e

# 每天早上 9 點執行（市場開盤後）
0 9 * * * /home/ec2-user/stock-underdog-ml/backtest/run_backtest.sh
```

### 查看分析結果

```bash
python backtest/analyze_backtest.py
```

---

## 📈 關鍵指標

### 1. 方向準確度（最重要）

預測漲跌方向的正確率：
- **>60%** - 優秀
- **50-60%** - 及格
- **<50%** - 需要改進

### 2. 平均絕對誤差

預測價格與實際價格的平均誤差：
- **<5%** - 非常準確
- **5-10%** - 可接受
- **>10%** - 需要優化

### 3. 誤差分布

- **誤差 ≤5%** - 高精度預測比例
- **誤差 ≤10%** - 中等精度預測比例

---

## 💾 資料庫查詢

### 查看模型整體表現

```sql
SELECT * FROM model_performance;
```

### 查看最近預測準確度

```sql
SELECT 
    ticker, 
    model_name,
    TO_CHAR(timestamp, 'YYYY-MM-DD') as pred_date,
    current_price,
    predicted_price,
    actual_price,
    percentage_error,
    CASE WHEN accuracy = 1.0 THEN '✅' ELSE '❌' END as direction
FROM predictions
WHERE actual_price IS NOT NULL
ORDER BY timestamp DESC
LIMIT 20;
```

### 查看待驗證預測

```sql
SELECT * FROM predictions_to_verify LIMIT 10;
```

---

## 🔧 進階設定

### 修改驗證頻率

編輯 `backtest.py` 第 65 行：
```python
max_search_days = 10  # 最多搜尋 10 天找下一交易日
```

### 過濾異常值

編輯 `analyze_backtest.py` 第 21 行：
```python
df = df[abs(df['percentage_error']) < 200]  # 過濾誤差 >200% 的異常值
```

---

## 📊 輸出範例

```
=== Next-Day Backtesting ===

1. Fetching predictions to verify...
Found 150 predictions without actual prices

2. Downloading next-day actual prices for 50 unique tickers...
  ✅ 2330.TW: Pred 1720.00, Actual 1715.00, Error 0.29%
  ✅ AAPL: Pred 195.00, Actual 196.50, Error -0.76%
  ❌ TSLA: Pred 250.00, Actual 240.00, Error 4.17%

✅ Updated 148 predictions
⏭️ Skipped 2 predictions (no next trading day data yet)

3. Model Performance Summary:

LSTM:
  Direction Accuracy: 58.5%
  Avg Absolute Error: 3.24%
  Total Verified: 148
```

---

## ❓ 常見問題

### Q: 為什麼有些預測沒有被驗證？

A: 可能原因：
1. 預測日期太近，下一交易日還沒到
2. 股票停牌或下市
3. 資料下載失敗

### Q: 方向準確度只有 55%，正常嗎？

A: 
- 55% 略高於隨機（50%），但還有改進空間
- 建議調整模型參數或增加特徵
- 參考 [../README.md](../README.md) 的模型優化章節

### Q: 可以改成預測 3 天後嗎？

A: 可以，修改 `models/lstm.py` 的 `predict_next_day()` 函數，
   但預測期間越長，準確度通常越低。

---

## 🔗 相關文檔

- [主要 README](../README.md) - 專案總覽
- [專案指南](../README.md) - 詳細安裝與使用說明
- [環境變數範本](../.env.example) - 設定參考

---

**💡 提示**: 建議每週檢視一次回測結果，持續優化模型參數。

