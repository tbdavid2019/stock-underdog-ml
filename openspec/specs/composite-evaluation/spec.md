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
The system SHALL aggregate Technical (XuanTie), ML (LSTM), and Institutional flow signals with Macro Regime exposure multipliers to compute final composite priority scores and identify Triple-Resonance candidates.

#### Scenario: Triple Resonance Qualification
- **WHEN** a stock concurrently satisfies XuanTie technical pullback, positive LSTM prediction, and positive institutional net accumulation
- **THEN** evaluator marks the stock with highest conviction priority and appends the `三重共振` badge

#### Scenario: Macro Exposure Discounting
- **WHEN** the US Macro Regime indicates Defensive or Panic status (exposure < 1.0)
- **THEN** evaluator applies exposure discount to final rank score and prepends macro risk warning badges to recommendations
