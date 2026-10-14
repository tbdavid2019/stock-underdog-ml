# Spec Delta: podcast-financial-radar

## ADDED Requirements

### Requirement: System One Decision Engine with 5-Tier Fallback
系統 SHALL 整合 System One 決策推論引擎，針對 Podcast 單集情報與提及標的進行型別化決策評估。推論鏈路 MUST 依序實施 5 級降級（Tier 1 Clef 主力 ➔ Tier 2 Clef 備援 ➔ Tier 3 JEV ➔ Tier 4 OpenAI ➔ Tier 5 本地啟發式規則）。若所有遠端決策模型均無法連線或逾時，系統 MUST 100% 降級為本地確定性規則，絕不中斷業務流程。

#### Scenario: Remote decision inference via Clef primary
- **WHEN** 系統發起單集或標的決策評估且 Clef 主力端點健康時
- **THEN** 系統以非自回歸方式取得市場重要性評分與標的多空機率分佈，標註 `source_slot: "clef_primary"`

#### Scenario: Complete remote failure degrades gracefully to local rules
- **WHEN** Clef 主力、Clef 備援、JEV 與 OpenAI 均斷線、逾時或驗證失敗時
- **THEN** 系統自動啟用本地確定性啟發式規則（依據頻道權重、來賓陣容、實體密度與情緒詞典），返回完整有效之決策與評級結構，標註 `source_slot: "local_rules"`，API 保持 HTTP 200 正常運作

### Requirement: Episode Importance and Market Impact Rating
系統 SHALL 依據訪談深度、講者權威性、實體討論強度與市場催化劑評估單集之市場重要性分數（1.0 至 5.0 分），並生成對應之視覺等級徽章（重磅 ⭐⭐⭐⭐⭐、重要 ⚡⚡⚡⚡、關注 📰📰📰）。

#### Scenario: Scoring high-catalyst episode
- **WHEN** 單集討論核心半導體供應鏈、央行決策或關鍵 AI 突破時
- **THEN** 系統評定重要性分數大於等於 4.0，並賦予重磅或重要之高優先級視覺徽章

#### Scenario: Filtering episodes by importance
- **WHEN** 使用者在終端點擊「重磅單集」篩選開關時
- **THEN** 介面即時過濾出重要性評分達標之關鍵情報，提升高價值資訊信噪比

### Requirement: Financial Entity and Ticker Sentiment Inference
系統 SHALL 針對單集內容中所聯結之上市公司股票代號，推論該單集對該標的之多空立場（`bullish` 看多、`bearish` 看空、`neutral` 中立），並計算推論確信度。

#### Scenario: Inferring bullish sentiment on mentioned stock
- **WHEN** 訪談內容對特定企業之財務指引、產品競爭力或訂單動能表達明確樂觀時
- **THEN** 系統判定該股票為 `bullish`，前端 Badge 呈現向上箭頭與看多色彩

#### Scenario: Inferring neutral or cautious sentiment
- **WHEN** 內容僅客觀引述背景或存在多空分歧時
- **THEN** 系統判定為 `neutral` 或 `bearish`，並提供相應之視覺辨識
