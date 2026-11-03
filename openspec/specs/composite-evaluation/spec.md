# composite-evaluation Specification

## Purpose
Combines multi-strategy signals, calculates composite factor scores, generates analytical tags, and ranks candidate stocks into prioritized recommendations with configurable overlap rules.

## Requirements

### Requirement: Cross-Strategy Composite Scoring and Ranking
The system SHALL aggregate arbitrary strategy outputs and fundamental valuation metrics using dynamic weights to calculate a normalized composite score (0-100) and rank stocks.

#### Scenario: Dynamic multi-strategy overlap detection
- **WHEN** a stock satisfies N or more enabled strategies (e.g. XuanTie + LSTM + Breakout)
- **THEN** evaluator marks stock as multi-strategy match and calculates composite priority score

#### Scenario: Dynamic weight configuration
- **WHEN** custom strategy weights are configured in evaluator settings
- **THEN** composite score is calculated as the normalized weighted sum of individual strategy scores and fundamental adjustments

#### Scenario: Fundamental factor scoring
- **WHEN** PE and PB are available and below valuation thresholds
- **THEN** evaluator appends positive fundamental score components and descriptive tags

### Requirement: Structured Evaluation Report Output
The system SHALL produce a structured evaluation report containing individual strategy breakdowns, multi-match intersection tables, and composite metrics formatted for console and notifier sinks.

#### Scenario: Report generation with dynamic strategies
- **WHEN** evaluation completes for an index with multiple enabled strategies
- **THEN** system outputs categorized sections for each strategy's hits, combined intersection candidates, and ranked recommendations

### Requirement: Triple Resonance and Macro Adjusted Ranking
The system SHALL aggregate Technical (XuanTie), Dual Foundation Models (Google TimesFM and Mistral Timestral), and Institutional flow signals with Macro Regime exposure multipliers to compute final composite priority scores and identify Multi-Dimensional Resonance candidates.

#### Scenario: Triple Resonance Qualification
- **WHEN** a stock concurrently satisfies XuanTie technical pullback, positive Google TimesFM prediction, positive Mistral Timestral P50 potential, and positive institutional net accumulation
- **THEN** evaluator marks the stock with highest conviction priority and appends the `🏆多維共振` badge

#### Scenario: Resonance Target Price Synthesis
- **WHEN** a stock qualifies for Multi-Dimensional Resonance
- **THEN** evaluator synthesizes the resonance target price as a weighted ensemble of TimesFM target price, Timestral P50 forecast, and technical resistance, rather than relying on a single model

#### Scenario: Macro Exposure Discounting
- **WHEN** the US Macro Regime indicates Defensive or Panic status (exposure < 1.0)
- **THEN** evaluator applies exposure discount to final rank score and prepends macro risk warning badges to recommendations

### Requirement: Explicit candidate promotion and health overlays
系統 SHALL 僅將已通過共同樣本及成本回測門檻、經管理設定晉升且用途資格成立的模型納入合成，以基礎權重乘健康倍率後正規化；shadow、disabled 或停用模型貢獻為零。預測漲幅排序與可信度排名須分開標示。

#### Scenario: Candidate promoted after evaluation
- **WHEN** MOIRAI-2 滿足晉升政策、用途資格且管理者啟用 active
- **THEN** 報告列出三模型版本、有效權重與合成來源，可逐筆重現合成價格

### Requirement: Truthful consensus labels
系統 SHALL 維持雙模型標籤僅指 TimesFM 3.0 與 Timestral 均為 active、可用、用途與健康政策允許且看漲；三模型共識須獨立標示。停用或單模型輸出須顯示實際參與者，模型目標須限定相同步長，2.5歷史共識須標示原版本。

#### Scenario: Original model suspended
- **WHEN** TimesFM 3.0 停用而 Timestral 與已晉升 MOIRAI-2 看漲
- **THEN** 顯示實際兩個參與模型，既有雙模型與四重共振資格不成立
