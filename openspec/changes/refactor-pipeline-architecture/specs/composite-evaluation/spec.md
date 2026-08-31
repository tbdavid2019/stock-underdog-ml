## Purpose

Combines multi-strategy signals, calculates composite factor scores, generates analytical tags, and ranks candidate stocks into prioritized recommendations with configurable overlap rules.

## ADDED Requirements

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
