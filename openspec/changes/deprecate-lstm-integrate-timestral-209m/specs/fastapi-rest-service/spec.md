# Spec Delta: fastapi-rest-service

## MODIFIED Requirements

### Requirement: Predictions Query Endpoints
The system SHALL provide REST endpoints to query quantitative predictions and strategy signals from DuckDB:
1. `GET /api/v1/predictions/latest`: Query latest snapshot by index with pagination.
2. `GET /api/v1/predictions/resonance`: Filter multi-strategy overlap and `🏆多維共振` candidates.
3. `GET /api/v1/predictions/xuantie`: Filter XuanTie MA pullback buy points.
4. `GET /api/v1/predictions/timestral/top-bullish`: Sort and retrieve top Mistral Timestral predicted gainers.
5. `GET /api/v1/predictions/timestral/top-bearish`: Sort and retrieve top Mistral Timestral predicted losers.
6. `GET /api/v1/predictions/history/{ticker}`: Retrieve time-series prediction trajectory.

#### Scenario: Querying latest resonance stocks
- **WHEN** client sends `GET /api/v1/predictions/resonance?min_hits=2`
- **THEN** system returns JSON list of candidates matching dual/triple resonance criteria with PE, PB, TimesFM, Timestral, and strategy tags.

#### Scenario: Querying Timestral top bullish predictions
- **WHEN** client sends `GET /api/v1/predictions/timestral/top-bullish?limit=20`
- **THEN** system returns top stocks sorted by Timestral forward potential percentage with P10/P50/P90 quantile data.

#### Scenario: Backward compatibility for legacy LSTM endpoints
- **WHEN** client sends `GET /api/v1/predictions/lstm/top-bullish` or `/api/v1/predictions/lstm/top-bearish`
- **THEN** system redirects or maps to Timestral foundation model prediction results with an HTTP response containing backward-compatible field aliases.
