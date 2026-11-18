# Spec Delta

## Purpose

聚合平台內建之時序基礎模型預測、台股法人與主力分點籌碼、Podcast 科技實體多空姿態與前沿技術雷達訊號，依確定性數值門檻合成具可追溯性之多源共識等級與背離風險警示。

## ADDED Requirements

### Requirement: Deterministic Multi-Source Signal Alignment Computation
系統 SHALL 依明確數值門檻將單一標的之時序模型潛力、籌碼淨流向、近 30 日 Podcast 實體立場與前沿雷達評分標準化為看多（`+1`）、中性（`0`）或看空（`-1`），並輸出一致看多、一致看空、訊號背離警示、多空分歧或來源不足分類。

#### Scenario: Bullish alignment across non-conflicting active sources
- **WHEN** 標的至少具備兩項有效適用訊號源為看多（`+1`）且無任何有效來源為看空（`-1`）
- **THEN** 系統輸出 `bullish_alignment`、實際有效來源數與分母，以及逐源標準化方向與依據

#### Scenario: Divergence warning between model forecast and flow or intelligence
- **WHEN** 時序模型維度與法人籌碼或情報維度出現一正（`+1`）一負（`-1`）之明確反向衝突
- **THEN** 系統輸出 `divergence_warning` 並標註衝突維度名稱，提醒使用者注意模型與籌碼或輿情背離

### Requirement: Market-Aware Null Preservation and Freshness Exclusion
系統 SHALL 對非台股之海外標的將台股法人籌碼標記為不適用（`null`）並自有效分母排除，絕不轉型為 `0`；對逾新鮮度視窗之來源標記 `stale` 並排除加權，若有效適用來源少於兩項則回傳 `insufficient_sources`。

#### Scenario: Overseas equity excludes Taiwan institutional dimension
- **WHEN** 使用者查詢美股或日股標的（如 `NVDA` 或 `7203.T`）之多源訊號一致度
- **THEN** 系統將籌碼維度標記為 `applicable: false` 且數值為 `null`，僅以模型、Podcast 與前沿雷達作為分母計算共識

#### Scenario: Insufficient valid sources after excluding stale data
- **WHEN** 扣除不適用與已逾 30 日新鮮度之過期情報後，標的僅剩單一有效訊號源
- **THEN** 系統標示過期項目為 `stale` 並回傳 `insufficient_sources`，不以單一維度偽造多源共識
