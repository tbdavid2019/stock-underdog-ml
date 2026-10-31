# pipeline-orchestrator Delta Specification

## ADDED Requirements

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
