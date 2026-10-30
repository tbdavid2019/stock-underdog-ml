# Spec Delta

## ADDED Requirements

### Requirement: Discovery Engine REST Endpoints
The system SHALL provide REST endpoints under `/api/v1/discovery` to serve discovery candidates, arXiv papers, news signals, history trends, and pipeline status with read-only database connections and cache acceleration.

#### Scenario: Query discovery candidates
- **WHEN** client sends `GET /api/v1/discovery/candidates`
- **THEN** system returns ranked candidate stocks with total_score breakdown, recommendation label, and supporting signals

#### Scenario: Query arXiv paper signals
- **WHEN** client sends `GET /api/v1/discovery/papers`
- **THEN** system returns list of recent technical papers with extracted bottlenecks and mapped equity tickers

#### Scenario: Query discovery history trends
- **WHEN** client sends `GET /api/v1/discovery/history`
- **THEN** system returns timeline data including candidate counts, paper counts, and signal counts across past runs

### Requirement: Discovery Health and Stale Status Reporting
The system SHALL provide `/api/v1/discovery/status` returning pipeline execution health, data freshness timestamp, and stale indicator.

#### Scenario: Report healthy fresh status
- **WHEN** pipeline completed successfully on current date
- **THEN** endpoint returns `status: "success"`, `stale: false`, and current `as_of` date

#### Scenario: Report stale fallback status
- **WHEN** pipeline encountered failure and retained previous snapshot
- **THEN** endpoint returns `status: "fallback"`, `stale: true`, and the previous valid snapshot timestamp
