## 1. 歷史殘留清理與核心配置 (Core & Cleanup)

- [ ] 1.1 實作 `core/device.py` 提供 `DeviceManager`（支援 CUDA / MPS / CPU 偵測與安全降級），並新增 `test/test_device.py` 驗證裝置解析與 fallback 機制
- [ ] 1.2 重構 `config.py` 為 `core/config.py`，移除 MySQL/MongoDB 廢棄變數，保留 Supabase、API 與 Notifier 配置並驗證設定載入正常
- [ ] 1.3 清理 `database.py` 中殘留的 MySQL 與 MongoDB 類別與函式，並確認 `test/test_json_safety.py` 與 `test/test_supabase_direct.py` 通過

## 2. 資料層與快取模組化 (Data Pipeline & Cache)

- [ ] 2.1 實作 `data/cache.py` 統一快取管理器（支援成分股 JSON 與股價 Pickle），並透過單元測試驗證 TTL 與清除功能
- [ ] 2.2 實作 `data/fundamentals.py` 提供 PE/PB/EV/EBITDA 基本面資料批次快取與 rate-limit 保護，並撰寫測試驗證欄位清洗與容錯
- [ ] 2.3 實作 `data/fetcher.py` 提供指數成分股載入（含 answerbook 失敗回退快取機制）與 OHLCV 批次下載，並撰寫測試驗證資料完整性

## 3. 策略標準化與綜合評價引擎 (Strategy & Evaluator)

- [ ] 3.1 定義 `strategies/base.py` 中的 `BaseStrategy`、`StockContext` 與 `StrategyResult` dataclass 契約
- [ ] 3.2 實作 `strategies/registry.py` 策略註冊中心（支援 `@register_strategy` 與動態策略工廠），並撰寫註冊機制測試
- [ ] 3.3 將玄鐵重劍策略封裝至 `strategies/xuantie.py`，並新增單元測試驗證買點信號與邊界條件
- [ ] 3.4 將 LSTM 預測模型封裝至 `strategies/lstm.py`，整合 `DeviceManager` 執行推論與訓練，並新增推論測試
- [ ] 3.5 實作 `evaluators/composite_evaluator.py` 進行動態多策略綜合評分、N-of-M 策略交集比對與標籤自動聚合，並撰寫加權排序測試
- [ ] 3.6 實作 `evaluators/formatter.py` 提供美化終端機表格與報告格式化

## 4. 管線排程與主程式重構 (Pipeline Orchestrator & Main)

- [ ] 4.1 實作 `pipeline/orchestrator.py` 實現兩階段管線（Stage 1 批次 I/O 預載 ➔ Stage 2 動態遍歷策略計算 ➔ Stage 3 綜合評分 ➔ Stage 4 儲存與通知）
- [ ] 4.2 重構 `main.py` 為輕量化入口，串接 `PipelineOrchestrator`，並移除過渡期檔案 `parallel_processor.py` 與 `main_lstm_only.py`
- [ ] 4.3 執行端到端執行測試（模擬或實際跑單一指數 TW50），驗證 console 輸出、Supabase 寫入與通知觸發均符合預期

## 5. 文件同步與發布驗收 (Documentation & Acceptance)

- [ ] 5.1 更新 `README.md`，更新系統架構圖、策略擴充指南、模組說明與環境變數說明
- [ ] 5.2 更新 `docs/CHANGELOG.md`，詳列本次重構項目、移除的廢棄模組與行為改善
