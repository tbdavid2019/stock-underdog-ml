## Why

目前量化選股專案隨歷次迭代累積了大量歷史包袱：
1. **職責混雜（God Main）**：`main.py` 同時包攬了網路請求、yfinance 呼叫、基本面抓取、模型訓練、多執行緒排程、字串硬編碼評分與結果通知。
2. **I/O 與運算未分離**：在 `ThreadPoolExecutor` 中同時做網路下載與模型訓練，受限於 Python GIL 且缺乏批次化（Batching）機制。
3. **缺乏硬體管理抽象**：未建構 CUDA / Apple Silicon MPS / CPU 自動探測與切換機制，相依套件同時載入未清理的框架。
4. **選股與評價未標準化**：各策略（玄鐵重劍、LSTM）回傳結構互不相容，綜合評價邏輯硬編碼在輸出迴圈中，無法靈活擴充新策略與權重調整。
5. **歷史殘留代碼**：`database.py` 與 `config.py` 仍保留早期未使用的 MySQL/MongoDB 代碼與設定。

現在進行系統性架構化重構，建立低耦合、高擴展性、支援硬體自動調度的現代化量化管線。

## What Changes

- **核心硬體管理 (`core-device-management`)**：新增 `DeviceManager`，提供 `cuda` / `mps` / `cpu` 自動偵測、手動指定與運行時安全降級。
- **資料抓取與快取層 (`stock-data-pipeline`)**：重構資料層，提供批次下載（Batch Fetch）、基本面資料批次快取、速率保護與統一快取策略。
- **策略引擎抽象化 (`stock-strategy-engine`)**：定義標準化 `BaseStrategy` 與 `StrategyResult` 契約，將「玄鐵重劍」與「ML 預測」封裝為獨立策略插件。
- **多維度綜合評價引擎 (`composite-evaluation`)**：獨立出 `CompositeEvaluator`，支援多策略權重分配、因子評分、標籤化及 Top-N 排序。
- **兩階段並行管線排程 (`pipeline-orchestrator`)**：將 Pipeline 拆解為「Stage 1: 批次 I/O 抓取」與「Stage 2: 向量化/批次運算」，精簡 `main.py` 成為乾淨的調度入口。
- **歷史廢碼清理**：移除 `database.py` 中的 MySQL/MongoDB 遺留代碼，清理無效的環境變數與重複的過渡期腳本。

## Capabilities

### New Capabilities
- `core-device-management`: 統一硬體運算設備管理（CUDA/MPS/CPU 偵測、配置與運行時上下文）。
- `stock-data-pipeline`: 統一資料抓取、批次下載、快取生命週期管理與基本面資料提供者。
- `stock-strategy-engine`: 標準化量化與機器學習策略介面（BaseStrategy、StrategyResult、信號與指標結構）。
- `composite-evaluation`: 多策略綜合評分器、多因子加權排序與標籤生成機制。
- `pipeline-orchestrator`: 兩階段（I/O vs Compute）並行排程器與輕量化主調度入口。

### Modified Capabilities
<!-- 無既有 specs 需要修改 -->

## Impact

- **程式碼架構**：新增 `core/`、`data/`、`strategies/`、`evaluators/`、`pipeline/` 模組，`main.py` 與 `database.py` 大幅精簡。
- **資料庫與相依性**：保留現有 Supabase `predictions` 寫入契約以維持向下相容；清理未使用的 `mysql-connector-python`、`pymongo` 殘留。
- **執行效能**：批次下載與 I/O/Compute 分流將大幅縮短每日掃描全市場（TW50, TW Mid 100, S&P 500）的執行時間。
