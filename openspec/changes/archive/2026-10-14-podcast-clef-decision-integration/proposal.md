# Proposal: Podcast Intelligence & Decision Model Integration with 5-Tier Resilient Fallback

## Why

目前系統已收錄全球頂級 100 大財經與科技 Podcast 頻道並具備基礎 TL;DR 萃取與股票代號標籤聯結。然而，面對每日海量單集情報，投資人無法迅速辨別**「哪些是重磅市場催化劑、哪些僅是泛泛閒聊」**，亦無法得知訪談內容對被提及股票之**具體多空立場（Bullish / Bearish / Neutral）**。

為此，我們需要導入極速非自回歸決策模型（Clef-Flash System One / JEV），為單集進行毫秒級的「市場重要性評級」與「標的多空態度推論」。更關鍵的是，依據架構鐵律：**「若所有遠端決策模型全部失敗或超時，系統必須能無縫退回本地啟發式規則（Local Rules Fallback），確保 100% 正常運作且永不中斷」**。

## What Changes

- **5 級高可用決策鏈 (5-Tier Resilient Decision Engine)**：
  - **Tier 1 (主力)**: `https://clef.create360.ai/v1/systemone` (Clef-Flash System One 快思考推理)
  - **Tier 2 (備援 1)**: `https://clef.aiurl.tw/v1/systemone` (llama.cpp Clef-Flash GGUF 節點)
  - **Tier 3 (備援 2)**: `https://api.typesafe.ai/v1/systemone` (JEV System One 原生端點)
  - **Tier 4 (備援 3)**: OpenAI 結構化決策模型 (Decisions / JSON Schema 降級)
  - **Tier 5 (兜底保障)**: **100% 本地確定性啟發式規則引擎 (Local Rule Heuristics Fallback)**。零外部網路依賴，依據頻道權重、來賓知名度、實體提及密度與關鍵字情緒詞典完成評級，保證任何上游掛掉時系統依然堅若磐石、秒級響應。
- **Podcast 單集深度智慧決策增強**：
  - **市場重要性評分 (`importance_score`, 1~5 分)** 與 **重要性徽章 (`importance_level`: 重磅 ⭐⭐⭐⭐⭐, 重要 ⚡⚡⚡⚡, 關注 📰📰📰)**。
  - **實體標的多空姿態 (`ticker_sentiments`)**：對被提及的上市公司 Ticker 自動判定 `bullish` (看多 ↑), `bearish` (看空 ↓), `neutral` (中立 ↔) 及其確信度。
- **後端架構與 DuckDB 持久化**：
  - `data/decision_service.py` 整合統一決策客戶端（支援個股量化決策與 Podcast 單集決策）。
  - `data/duckdb_manager.py` 擴充 `podcast_episodes` 儲存欄位或 JSON 負載（`importance_score`, `importance_level`, `ticker_sentiments`, `decision_slot`）。
  - `api/routes/podcast.py` 新增單集與批量決策評估端點 `POST /api/v1/podcast/episodes/{episode_id}/evaluate`、`POST /api/v1/podcast/evaluate-batch`。
- **前端介面升級 (Impeccable 工藝標準)**：
  - 單集卡片右上角標註 AI 重要性徽章（深淺主題自適應）。
  - 股票代號 Badge 依據決策結果呈現動態多空指示（綠色看漲、紅色看跌、灰色中立）。
  - 頂部話題列支援「重磅單集」一鍵快篩。

## Capabilities

### Modified Capabilities
- `podcast-financial-radar`: 擴充 Podcast 財經雷達之 AI 決策與多空推論規範，並定義 5 級降級防護（含本地啟發式規則兜底）。

## Impact

- **後端模組**：
  - `data/decision_service.py`（新模組，整合 Clef/Jev/OpenAI/LocalRules 5 級決策鏈）
  - `data/clef_client.py`（與 decision_service 整合相容）
  - `data/podcast_service.py`（導入決策評估與即時單集增強）
  - `data/duckdb_manager.py`（更新單集持久化欄位）
  - `api/routes/podcast.py`（新增 evaluation 路由）
  - `api/routes/decision.py`（擴充 health 與統一調用）
- **前端模組**：
  - `api/templates/partials/tab_podcast.html`（重要性標籤、多空箭頭）
  - `api/static/js/app.js`（狀態綁定與過濾邏輯）
- **相依性**：無新增重大相依套件，沿用標準 `requests`、`json`、`duckdb`。
