# Scripts Directory

測試、除錯、維護和遷移腳本集合。

## 📁 目錄結構

```
scripts/
├── migration/                           🆕 數據庫遷移腳本
│   ├── supabase_add_columns.sql              為現有表格添加雙軌策略欄位
│   ├── supabase_drop_dual_table.sql          刪除測試表格
│   └── add_columns_to_predictions.py         遷移幫助腳本
│
├── verification/                        🆕 驗證腳本
│   └── verify_predictions_extended.py        驗證表格結構
│
├── test_next_day_prediction.py              測試下一日預測邏輯
├── backfill_historical.py                   回填歷史預測的實際價格
├── check_db_contents.py                     檢查資料庫內容
├── check_cache.py                           檢查快取狀態
├── test_cache.py                            測試快取功能
├── debug_backfill.py                        除錯回測問題
├── clear_backtest_data.py                   清空回測資料
├── backtest_analysis.py                     回測分析
├── app.py                                   Streamlit 測試應用
└── README.md                                本文件
```

## 🆕 遷移腳本 (Migration)

### 用途
為現有用戶升級到雙軌策略系統。

### 使用時機
- 從舊版本（純 LSTM）升級到新版本（LSTM + 玄鐵策略）
- 需要在現有 `predictions` 表格添加新欄位

### 執行方式
```bash
# 1. 在 Supabase SQL Editor 執行
#    scripts/migration/supabase_add_columns.sql

# 2. 驗證
python scripts/verification/verify_predictions_extended.py

# 3. (可選) 清理測試表
#    在 Supabase SQL Editor 執行
#    scripts/migration/supabase_drop_dual_table.sql
```

**注意**: 新用戶直接執行根目錄的 `supabase_schema.sql`，無需執行 migration 腳本。

---

## ✅ 驗證腳本 (Verification)

### verify_predictions_extended.py
驗證 `predictions` 表格包含所有雙軌策略欄位：
- strategy_type
- ma5, ma10, ma60, ma120, ma250
- pullback_type
- pe, pb, forward_pe

```bash
python scripts/verification/verify_predictions_extended.py
```

---

## 🧪 測試腳本

### 預測相關
- **test_next_day_prediction.py** - 測試 LSTM 下一日預測邏輯
- **test_cache.py** - 測試股票數據快取功能

### 數據庫相關
- **check_db_contents.py** - 查看資料庫內容和統計
- **check_cache.py** - 檢查快取狀態和有效期

### 回測相關
- **backfill_historical.py** - 回填歷史預測的實際價格用於驗證
- **backtest_analysis.py** - 分析回測結果和模型表現
- **debug_backfill.py** - Debug 回填過程的問題
- **clear_backtest_data.py** - 清空回測數據重新開始

### 其他
- **app.py** - Streamlit 網頁測試應用（可視化預測結果）

---

## 📝 常用命令

```bash
# 檢查資料庫
python scripts/check_db_contents.py

# 測試預測功能
python scripts/test_next_day_prediction.py

# 回填並分析回測
python scripts/backfill_historical.py
python scripts/backtest_analysis.py

# 檢查快取
python scripts/check_cache.py
```

---

## 🔗 相關文件

- [../supabase_schema.sql](../supabase_schema.sql) - 完整資料庫結構（新用戶使用）
- [../README.md](../README.md) - 專案主文件
- [../main.py](../main.py) - 主程式入口（雙軌策略）
- [../main_lstm_only.py](../main_lstm_only.py) - 單純 LSTM 版本（舊版保留）
