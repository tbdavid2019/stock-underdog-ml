# Design: Podcast Intelligence & Decision Model Integration with 5-Tier Resilient Fallback

## Context

當前系統已在 `data/clef_client.py` 具備初步的 Clef System One 調用架構，並在 `data/podcast_service.py` 建立了 100 個主流 Podcast 頻道與種子單集情報庫。然而，單集情報僅有文字 TL;DR 與靜態 Ticker 聯結，缺乏量化決策模型賦予的「市場催化劑重要性」與「標的多空態度推論」。

使用者明確指示：
1. 導入 Clef / JEV / OpenAI 決策模型為 Podcast 標籤與重要性評估賦能。
2. **最核心架構約束**：**「若決策模型全部都失敗，也要退回可以運作」**。

## Goals / Non-Goals

**Goals:**
- **5 級階梯式韌性決策鏈 (5-Tier Resilient Decision Fallback Chain)**：
  - Tier 1: `https://clef.create360.ai/v1/systemone`（主力 Clef-Flash System One 節點）
  - Tier 2: `https://clef.aiurl.tw/v1/systemone`（備援 Clef-Flash llama.cpp 節點）
  - Tier 3: `https://api.typesafe.ai/v1/systemone`（JEV 決策模型原生端點）
  - Tier 4: OpenAI 結構化決策模型（Decisions / JSON Schema 降級路徑）
  - Tier 5: **100% 本地確定性啟發式規則引擎 (Deterministic Local Rule Heuristics)**。
- **永不崩潰保證 (Zero Downtime / Zero Crash Guarantee)**：當所有遠端模型（Tier 1~4）因網路斷線、503 服務不可用、逾時或格式錯誤時，系統必須在 5ms 內無縫切換至本地啟發式規則，返回型別完全一致的有效決策數據，HTTP 永遠維持 200 OK。
- **Podcast 單集與 Ticker 雙重賦能**：
  - 單集層級：市場重要性評級（`importance_score` 1.0~5.0 分、`importance_level` 重磅/重要/關注）。
  - 實體標的層級：針對被提及的股票代號（如 `NVDA`, `2330.TW`）推論 `bullish` (看多 ↑), `bearish` (看空 ↓), `neutral` (中性 ↔) 姿態與確信度。
- **DuckDB 高速持久化與快取**：評估結果即時寫入 DuckDB `podcast_episodes` 表，避免重複消耗推論算力。
- **前端高工藝展示**：Material Symbols 圖標、深淺雙主題自適應徽章、股票多空動態顏色指示。

**Non-Goals:**
- 不取代 Stage 2 的多策略量化模型（玄鐵均線、LSTM、TimesFM 仍然是核心選股硬指標）；System One 決策模型作為共振加權與信號過濾器。
- 不進行長篇逐字稿的串流音訊即時轉錄（沿用現有 RSS / Transcript 提煉流）。

## Decisions

### 1. 統一決策服務架構 (`data/decision_service.py`)
建立統管個股與 Podcast 的雙工決策中樞 `DecisionService`，封裝 5 級降級鏈路：
- 支援 `evaluate_stock_decision()` 與 `evaluate_podcast_decision()`。
- 單一端點調用逾時嚴格設為 3 秒（防止串聯等待過久），總等待時間上限控制在 10 秒內。
- 每個決策物件明確標註 `source_slot`（`clef_primary` / `clef_aiurl` / `jev` / `openai` / `local_rules`），維持高度透明性與可觀測性。

### 2. Podcast 決策專用 State 與 Questions 合約規範
- **State 結構**：
  ```json
  {
    "channel_title": "All-In Podcast",
    "episode_title": "E150: AI chip race and macro liquidity",
    "duration_minutes": 75,
    "host": "Chamath Palihapitiya, Jason Calacanis",
    "guest": "Brad Gerstner",
    "tldr": "...",
    "mentioned_tickers": ["NVDA", "TSMC", "MSFT"]
  }
  ```
- **Questions 定義**：
  - `importance_score`: `score` 題型（1~5 級， criteria: 1_低關注 ~ 5_重磅催化劑）。
  - `market_catalyst`: `choice` 題型（criteria: `high_impact`, `sector_trend`, `macro_policy`, `general_discussion`）。
  - `ticker_sentiments`: 動態依據 `mentioned_tickers` 產生子評估（choice: `bullish`, `bearish`, `neutral`）。

### 3. 本地啟發式規則兜底引擎 (Tier 5 Fallback)
若 Tier 1~4 全數失效，本地規則引擎以零依賴的確定性演算法計算：
- **基礎重要性評分**：
  - 頻道權重：國際一線頻道（Bloomberg, All-In, Lex Fridman, Acquired, 股癌等）基礎分 3.5 分。
  - 來賓加成：若 Guest 包含知名 CEO/創辦人/經濟學家，加 0.5~0.8 分。
  - 實體密度：提及 3 個以上上市公司或核心科技實體，加 0.5 分。
  - 封頂 5.0 分，底限 1.0 分。
- **標的多空立場評定**：
  - 透過中英文金融情感關鍵字詞庫（看多詞：突破、買進、看好、擴產、超預期、新高、bullish；看空詞：跌破、衰退、砍單、下修、利空、裁員、bearish）。
  - 在 Ticker 出現前後 100 字元滑動窗口內計算多空情感權重，若無明顯傾向則歸入 `neutral`。

### 4. 前端極致體驗與 RWD 適配
- 單集情報卡右上角新增優雅的星級膠囊（如 `⭐ 4.5 重磅`、`⚡ 3.8 重要`）。
- 股票代碼 Badge（如 `NVDA`）後綴小箭頭（`↑ 看多`、`↓ 看空`、`↔ 持平`），並賦予主題感應色彩。

## Risks / Trade-offs

- **[風險 1] 遠端 4 級模型若連續失敗，串聯超時可能拉長 API 延遲**  
  ➔ **緩解**：每個遠端 Slot 逾時控制在 2.5~3 秒；若偵測到斷線（如 ConnectError），立即短路跳至下一級，確保就算全部掛掉，總延遲也不超過 6 秒，並由本地規則迅速完成。
- **[風險 2] OpenAI API 缺少 Key 或餘額不足**  
  ➔ **緩解**：若環境變數未配置 `OPENAI_API_KEY`，自動跳過 Tier 4，平滑降級至 Tier 5 本地規則，完全不報錯。
- **[風險 3] DuckDB 舊資料可能無 `importance_score` 欄位**  
  ➔ **緩解**：在 `data/duckdb_manager.py` 實作自動 schema migration（`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`），若無資料則安全 fallback，保障歷史數據相容性。
