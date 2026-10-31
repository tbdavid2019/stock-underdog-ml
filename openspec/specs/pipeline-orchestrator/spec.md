# pipeline-orchestrator Specification

## Purpose
Orchestrates the end-to-end execution pipeline by separating concurrent network I/O retrieval from compute workloads and routing results to persistence and notification sinks.

## Requirements

### Requirement: Two-Stage Execution Pipeline
The system SHALL organize index analysis into two distinct stages: Stage 1 for concurrent I/O data prefetching and Stage 2 for compute-bound strategy execution and evaluation.

#### Scenario: Stage 1 prefetching completion
- **WHEN** an index component list is received
- **THEN** orchestrator concurrently downloads and caches all required market prices and fundamentals before strategy computation begins

#### Scenario: Stage 2 compute execution
- **WHEN** all data is loaded in memory/cache
- **THEN** orchestrator passes preloaded datasets to strategies for vectorized or parallel computation

### Requirement: Sinks Integration and Resiliency
The system SHALL route evaluation results to Supabase database and notification channels with comprehensive error isolation.

#### Scenario: Database sink failure isolation
- **WHEN** Supabase persistence encounters a network or schema error
- **THEN** system logs the error and continues to output console reports and send notifications without crashing the entire run

### Requirement: Macro Pre-flight Check and Multi-Source Enrichment
The system SHALL execute a pre-flight macro regime check prior to index batch processing and enrich notifications and database persistence with macro state and institutional metrics.

#### Scenario: Pre-flight Macro Check Execution
- **WHEN** the orchestrator initiates daily multi-index analysis
- **THEN** it downloads macro tickers (`SPY`, `^VIX`, `^SOX`), computes market regime and exposure level, and attaches the macro summary to the top of all output reports

#### Scenario: Enriched Persistence and Push Alerts
- **WHEN** candidate predictions are persisted to Supabase and dispatched via Telegram/Discord
- **THEN** the payload includes macro exposure status and 5-day foreign/investment trust net metrics without breaking existing schemas

### Requirement: Scaled Market Routing and Segmented Partitioning
The system SHALL support 1,000 Taiwan stocks and 500 US stocks in daily quant forecasting, with partition support to preserve telemetry, reduce duplicate computations, and respect notification limits.

#### Scenario: Taiwan market execution partitioning
- **WHEN** `--market tw` is executed
- **THEN** orchestrator executes analysis across `台灣50` (50), `台灣中型100` (100), and `台股活躍精選` (850 disjoint active/OTC champions), covering a total of 1,000 unique stocks.

#### Scenario: Monolithic Taiwan 1000 execution
- **WHEN** `--index 台股1000` or `--index TW1000` is requested
- **THEN** orchestrator executes analysis across the single consolidated 1,000-stock universe.

#### Scenario: US market full scale execution
- **WHEN** `--market us` is executed
- **THEN** orchestrator executes analysis across the full `SP500` pool of 500+ stocks.
