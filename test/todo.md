# Stock Application Refactoring To-Do List

Based on the analysis of `app.py`, the following improvements are recommended:

## 1. 程式架構與模組化 (Structure & Modularity)
目前 `app.py` 過於龐大（近 1000 行），建議拆分為以下模組：
- [x] **config.py**: 集中管理 `.env` 環境變數與常數設定。
- [x] **database.py**: 封裝 `MySQLManager` 和 MongoDB 連線邏輯，實作連線管理。
- [x] **data_loader.py**: 負責 API 爬蟲（stock lists）與 `yfinance` 數據抓取。
- [x] **models/**: 將各模型邏輯拆分為獨立檔案：
    - `models/lstm.py`
    - `models/transformer.py`
    - `models/prophet_model.py`
    - `models/cross_section.py` (TabNet, etc.)
- [x] **notifier.py**: 封裝 Email, Telegram, Discord 的發送邏輯。
- [x] **main.py**: 簡化為主流程控制腳本。

## 2. 執行效能 (Performance)
- [ ] **實作並行處理 (Parallel Processing)**:
    - 使用 `concurrent.futures.ThreadPoolExecutor` 或 `ProcessPoolExecutor` 來並行處理多支股票的訓練與預測，解決 S&P 500 等大量股票運算過久的問題。
- [ ] **優化資料下載**:
    - 在主流程開始時統一批次下載所需資料（使用 `yfinance` 的 batch 下載功能），避免在迴圈中重複請求。
- [ ] **模型訓練策略優化**:
    - 評估是否需要對每支股票每次都重新訓練 (`fit`)，或可改用增量訓練 / 通用模型策略。

## 3. 資料庫遷移至 Supabase (Database Migration)
- [ ] **統一遷移至 Supabase**:
    - 將 MySQL 和 MongoDB 統一遷移至 Supabase (PostgreSQL)
    - 使用 Supabase 的即時資料庫功能，同時滿足關聯式查詢與 NoSQL 彈性需求
- [ ] **認證設定**:
    - 在 `.env` 中新增 `SUPABASE_URL` 和 `SUPABASE_SERVICE_ROLE_KEY`
    - 安裝 `supabase-py` 套件：`pip install supabase`
- [ ] **重構資料庫模組**:
    - 建立 `database.py` 封裝 Supabase 連線邏輯
    - 實作 Context Manager 進行連線管理
    - 設計資料表 Schema (predictions, stock_data 等)
- [ ] **資料遷移**:
    - 匯出現有 MySQL/MongoDB 資料
    - 在 Supabase 建立對應的 Tables
    - 執行資料遷移腳本
- [ ] **連線優化**:
    - 使用 Supabase 的內建 Connection Pool
    - 實作錯誤重試機制

## 4. 錯誤處理與日誌 (Error Handling & Logging)
- [ ] **導入 Logging 模組**:
    - 替換所有的 `print()` 為 Python `logging`，設定適當的 Log Level (INFO, ERROR, DEBUG)。
    - 將 Log 輸出至檔案以便追蹤。
- [ ] **例外彙整**:
    - 在批次處理中捕捉失敗的股票代碼，並在最後的通知內容中加入「失敗清單」與錯誤摘要。

## 5. 常數與設定管理
- [ ] **移除 Hardcoded URLs**: 將 `get_tw0050_stocks` 等函式中的 URL 移至設定檔。
- [ ] **資料洩漏檢查**: 修正 `prepare_data` 中的 Scaler 邏輯，確保只使用 Training Data 進行 `fit`。
