# tech-radar-discovery-engine Specification

## Purpose
提供前沿學術論文與產業快訊的主動發現引擎，透過語意解析提取技術瓶頸，並結合 7 維度多因子評分模型輸出候選標的推薦與歷史趨勢追蹤。

## Requirements

### Requirement: Academic Paper Signal Ingestion
The system SHALL ingest academic papers from arXiv (categories: cs.AI, cs.RO, cs.LG, cs.CV) with a 3-second request throttle and exponential backoff retry for HTTP 429 rate limits.

#### Scenario: Ingest arXiv daily feed successfully
- **WHEN** the discovery pipeline executes daily collection
- **THEN** latest arXiv papers are parsed into structured paper records with abstracts, primary categories, and bottleneck metadata

#### Scenario: Graceful fallback on upstream network timeout
- **WHEN** arXiv API times out or exceeds retry limits
- **THEN** the system logs a warning, retains the previous day's paper cache, and marks discovery health status as stale fallback

### Requirement: Ticker Canonicalization and Universe Whitelist Validation
The system SHALL canonicalize equity tickers and strictly filter candidate stocks against the verified stock universe whitelist before scoring.

#### Scenario: Canonicalize Taiwan and US tickers
- **WHEN** candidate tickers include variations such as `2330.TW` or lowercase `nvda`
- **THEN** the canonicalizer normalizes them to `2330` and `NVDA`

#### Scenario: Filter out non-universe tickers
- **WHEN** an extracted ticker does not exist in the active stock universe whitelist
- **THEN** the candidate is discarded from discovery scoring and recorded to rejected signal logs

### Requirement: Deterministic Seven-Factor Discovery Scoring
The system SHALL compute a deterministic composite score clamped to [0, 100] across 7 factors and classify candidates into four recommendation categories.

#### Scenario: High conviction candidate categorized as propose_add
- **WHEN** an un-tracked stock achieves total_score >= 75 and purity_score >= 60
- **THEN** the candidate is labeled as `propose_add` with review_status `pending`

#### Scenario: Existing tracked stock categorized as already_in_pool
- **WHEN** a stock already present in the active tracking pool receives reinforced signals
- **THEN** the candidate is labeled as `already_in_pool` with cross-signal references

### Requirement: Two-Phase Atomic Persistence and Data Retention
The system SHALL execute two-phase atomic persistence with DuckDB transactions followed by atomic JSON cache replacement, and prune records older than the TTL policy.

#### Scenario: Atomic cache generation after database commit
- **WHEN** scoring pipeline completes and commits DuckDB transaction
- **THEN** cache files are written to temporary files and atomically replaced via file system replace

#### Scenario: Apply TTL pruning to historical records
- **WHEN** pipeline finishes daily snapshot generation
- **THEN** papers older than 180 days and signals older than 90 days are deleted from DuckDB
