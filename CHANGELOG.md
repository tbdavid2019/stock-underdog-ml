# Changelog

所有重要的專案變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)。

---

## [2.1.0] - 2026-09-01

### Added

- **模組化分層架構 (Modular Pipeline Architecture)**：新增 `core/`、`data/`、`strategies/`、`evaluators/`、`pipeline/` 模組，徹底解決單一程式碼檔案過度耦合（God Object）的問題。
- **統一硬體管理 (`core.device.DeviceManager`)**：支援 CUDA、Apple Silicon (MPS) 與 CPU 運行時自動偵測、手動指定與平滑降級。
- **統一快取與資料管線 (`data.cache.CacheManager`, `data.fetcher.StockFetcher`, `data.fundamentals.FundamentalProvider`)**：
  - 支援指數成分股 JSON 快取與行情 Pickle 快取生命週期管理。
  - 基本面估值指標（PE/PB/EV/EBITDA）批次快取與錯誤容錯。
  - 完整保留 `answerbook` 優先最新、失敗回退本機快取之可靠性策略。
- **美股與太空概念標的擴充**：在 S&P500 / 美股分析清單中強制納入 SpaceX 相關標的（`SPCX`、`DXYZ`、`ASTS`、`RKLB`），擴大太空衛星概念選股覆蓋度。
- **插件式策略註冊中心 (`strategies.registry.StrategyRegistry`)**：
  - 標準化 `BaseStrategy`、`StockContext` 與 `StrategyResult` 契約。
  - 提供 `@register_strategy` 裝飾器，使未來擴充技術面、籌碼面、機器學習選股策略只需新增單一檔案。
  - 完整封裝 `XuanTieStrategy`（玄鐵重劍）與 `LSTMStrategy`（LSTM 深度學習）。
- **動態多策略綜合評價引擎 (`evaluators.composite_evaluator.CompositeEvaluator`)**：
  - 支援多策略動態權重評分（0~100）與 N-of-M 多策略交集自動標籤聚合。
  - 獨立的美化終端機表格輸出工具 (`evaluators.formatter.print_evaluation_report`)。
- **兩階段執行調度器 (`pipeline.orchestrator.PipelineOrchestrator`)**：
  - Stage 1 (I/O Prefetch) 與 Stage 2 (Compute) 分流，大幅改善並發效率並避免 GIL 衝突。

### Changed

- `answerbook` 指數榜單每次執行優先抓取最新資料；只有上游 API 失敗時才使用未超過 90 天的快取。
- 保留 S&P500 策略前 110 名限制，但每次先取得最新的完整榜單再取前 110 名。

### Fixed

- Supabase 寫入前新增嚴格 JSON 檢查器，將 `NaN` 與無限大數值轉為 `null`，其他不可序列化值則取消寫入，避免整批資料因 JSON 格式失敗。
- 修正 TensorFlow/Keras 在模組載入階段過早導入導致 `curl_cffi` 產生 `ImpersonateError` TLS 握手衝突的底層問題，全面改為 Stage 2 延遲載入（Lazy-load），確保 yfinance 批次行情下載 100% 成功。

### Documentation

- 新增 `AGENTS.md`，規範 changelog、README、安全與測試同步要求。

### Maintenance

- 將 `main.py` 重構為輕量化管線調度入口（縮減至 40 行左右）。
- 清理 `database.py` 中殘留的 MySQL 與 MongoDB 廢棄類別與驅動檢查。
- 將過渡期檔案 `parallel_processor.py` 與 `main_lstm_only.py` 移至 `archive/`。
- `run_daily.sh` 補充 `PYTHONPATH` 與系統 `PATH` 環境變數匯出，確保在 Cron 最小環境下順利執行回測與主程式。
- 於 `10.9.0.99` 部署自動化 Cron 排程：包含台股收盤後（15:30 CST / 07:30 UTC）與美股收盤後（07:00 CST / 23:00 UTC）自動執行雙軌選股分析。

## [2.0.0] - 2026-01-08

### 🎯 重大變更

#### 預測邏輯重構
- **改為預測下一交易日** - 從預測「歷史資料最大值」改為預測「下一個交易日的收盤價」
- **實用性大幅提升** - 現在可以直接用於交易決策參考

#### 模型精簡
基於回測結果（543 筆資料），關閉表現不佳的模型：

| 模型 | 狀態 | 方向準確度 | 原因 |
|------|------|-----------|------|
| **LSTM** | ✅ 保留 | 54.6% | 略高於隨機，有優化空間 |
| **Prophet** | ❌ 關閉 | 46.9% | 低於隨機，不適合短期預測 |
| **Chronos** | ❌ 關閉 | N/A | 執行失敗，模型太大太慢 |
| **CROSS** | ❌ 關閉 | N/A | 執行失敗 |
| **Transformer** | ❌ 關閉 | N/A | 資源消耗大 |

**結論：** 專注優化 LSTM，目標提升方向準確度至 60%+

### ✨ 新增功能

- **自動回測系統** - 每日驗證預測準確度
  - 自動抓取下一交易日的實際股價
  - 計算方向準確度、平均誤差等指標
  - 支援按指數分析（台灣50、台灣中型100、SP500）
  
- **回測腳本** - `backtest/backtest.py`
  - 智能處理交易日（自動跳過週末和假日）
  - 批次處理，約 2.6 分鐘處理 1000 筆預測
  
- **分析工具** - `backtest/analyze_backtest.py`
  - 詳細的模型表現分析
  - 按模型和指數分組統計

### 🔧 優化改進

#### LSTM 模型優化 1/9/2026 生效
- **網路深度增加** - 從 2 層增加到 3 層 LSTM（128→64→32 units）
- **訓練時間延長** - epochs 從 10 增加到 100
- **Early Stopping** - 防止過擬合
- **Learning Rate Scheduling** - 動態調整學習率
- **時間窗口增加** - time_step 從 60 增加到 90 天
- **批次大小優化** - 從 32 降到 16（更細緻的梯度更新）

#### 專案結構優化
- **新增 `scripts/` 資料夾** - 整理測試和維護腳本
- **新增 `backtest/` 資料夾** - 回測系統獨立管理
- **根目錄精簡** - 只保留核心程式檔案

### 🐛 Bug 修復

- **修復資料洩漏** - `MinMaxScaler` 現在只在訓練資料上 fit
- **修復日期比對錯誤** - 回測時正確處理 `datetime.date` vs `datetime.datetime`
- **修復 Supabase 連線** - 優先使用 `SUPABASE_SERVICE_KEY`

### 🔒 安全性

- **更新 `.gitignore`** - 防止敏感資訊被提交
  - 新增 `.env*`, `*.env.backup` 等模式
  - 清理 git 歷史中的 `.env.backup`

### 📝 文檔更新

- **README.md** - 全面改寫，聚焦 LSTM 和下一日預測
- **SETUP.md** - 新增完整安裝指南
- **backtest/README.md** - 回測系統詳細說明
- **scripts/README.md** - 測試腳本使用說明
- **CHANGELOG.md** - 本檔案

### 📊 回測數據

基於 2026-01-05 ~ 2026-01-08 的 543 筆預測：

**LSTM 表現：**
- 方向準確度：54.6%
- 平均絕對誤差：21.3%
- 誤差 ≤10%：53.3%
- 誤差 ≤20%：71.2%

**各指數表現：**
- 台灣中型100：51.9%（最佳）
- SP500：48.5%
- 台灣50：41.5%（需改進）

### ⚙️ 設定變更

**推薦的 `.env` 設定：**
```bash
USE_PROPHET=false
USE_CHRONOS=false
USE_CROSS=false
USE_TRANSFORMER=false
```

### 🗑️ 移除項目

- Prophet 模型支援（可透過 `USE_PROPHET=true` 重新啟用，但不建議）
- Chronos 模型支援（可透過 `USE_CHRONOS=true` 重新啟用，但不建議）
- MySQL 資料庫支援（已完全遷移至 Supabase）

---

## [1.0.0] - 2026-01-05

### 初始版本

- 支援多種模型：LSTM, Prophet, Chronos, Transformer, Cross-sectional
- MySQL 資料庫儲存
- Discord, Telegram, Email 通知
- 多指數支援：台灣50, 台灣中型100, SP500, NASDAQ 等

---

## 未來計劃

### v2.1（計劃中）
- [ ] LSTM 準確度優化至 60%+
- [ ] 新增更多技術指標作為特徵
- [ ] 實作模型 ensemble
- [ ] 新增 Web UI 查看預測結果

### v3.0（構思中）
- [ ] 支援盤中即時預測
- [ ] 新增風險評估指標
- [ ] 整合基本面數據
- [ ] 支援自動交易（紙上交易）

---

**注意：** 本專案僅供學習和研究使用，不構成投資建議。
