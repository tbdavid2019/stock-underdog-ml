# Scripts Directory

測試、除錯和維護腳本集合。

## 📁 目錄結構

```
scripts/
├── README.md                    ← 你在這裡
├── test_next_day_prediction.py  ← 測試下一日預測邏輯
├── backfill_historical.py       ← 回填歷史預測的實際價格
├── check_db_contents.py         ← 檢查資料庫內容
├── debug_backfill.py            ← 除錯回測問題
├── clear_backtest_data.py       ← 清空回測資料
├── app.py                       ← 舊版主程式（已棄用）
└── backtest_analysis.py         ← 舊版回測分析（已棄用）
```

## 🔧 腳本說明

### test_next_day_prediction.py
測試新的下一日預測邏輯。

**用途：** 在部署前驗證預測功能是否正常

**執行：**
```bash
python scripts/test_next_day_prediction.py
```

**輸出：** 3 支測試股票的預測結果

---

### backfill_historical.py
回填歷史預測的實際價格並計算準確度。

**用途：** 批次更新所有未驗證的預測

**執行：**
```bash
python scripts/backfill_historical.py
```

**輸出：**
- 更新數量
- 各指數準確度統計
- 執行時間

---

### check_db_contents.py
檢查 Supabase 資料庫的預測記錄。

**用途：** 快速查看資料庫狀態

**執行：**
```bash
python scripts/check_db_contents.py
```

**輸出：**
- 總預測數
- 日期範圍
- 回測狀態
- 樣本資料

---

### debug_backfill.py
除錯回測腳本的問題。

**用途：** 當回測失敗時，診斷原因

**執行：**
```bash
python scripts/debug_backfill.py
```

**輸出：** 詳細的日期比對和資料下載資訊

---

### clear_backtest_data.py
清空所有回測欄位（保留預測記錄）。

**用途：** 重置回測資料，重新驗證

**執行：**
```bash
python scripts/clear_backtest_data.py
```

**警告：** 會清空所有 `actual_price` 等欄位

---

## 🔗 相關文檔

- [主要 README](../README.md) - 專案總覽
- [回測系統](../backtest/README.md) - 回測說明
- [安裝指南](../SETUP.md) - 環境設定

---

## 💡 使用建議

1. **部署前測試：** 執行 `test_next_day_prediction.py`
2. **定期回填：** 每週執行 `backfill_historical.py`
3. **問題診斷：** 使用 `check_db_contents.py` 和 `debug_backfill.py`
4. **重置資料：** 謹慎使用 `clear_backtest_data.py`
