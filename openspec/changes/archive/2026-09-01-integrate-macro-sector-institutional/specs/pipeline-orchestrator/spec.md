## ADDED Requirements

### Requirement: Macro Pre-flight Check and Multi-Source Enrichment
The system SHALL execute a pre-flight macro regime check prior to index batch processing and enrich notifications and database persistence with macro state and institutional metrics.

#### Scenario: Pre-flight Macro Check Execution
- **WHEN** the orchestrator initiates daily multi-index analysis
- **THEN** it downloads macro tickers (`SPY`, `^VIX`, `^SOX`), computes market regime and exposure level, and attaches the macro summary to the top of all output reports

#### Scenario: Enriched Persistence and Push Alerts
- **WHEN** candidate predictions are persisted to Supabase and dispatched via Telegram/Discord
- **THEN** the payload includes macro exposure status and 5-day foreign/investment trust net metrics without breaking existing schemas
