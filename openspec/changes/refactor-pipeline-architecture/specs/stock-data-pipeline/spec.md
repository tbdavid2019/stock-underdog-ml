## Purpose

Manages stock index components, historical market OHLCV price series, and fundamental metrics retrieval with multi-tiered caching and batch operations.

## ADDED Requirements

### Requirement: Unified Market Data Fetching and Caching
The system SHALL retrieve historical stock price data for single and multiple tickers, using disk caching with configurable expiration policies and upstream fallback.

#### Scenario: Cache hit for stock data
- **WHEN** valid unexpired cached data exists for a ticker and period
- **THEN** system loads data from cache without sending remote network requests

#### Scenario: Cache miss and download
- **WHEN** cache is missing or expired
- **THEN** system downloads data from upstream provider and writes updated data to cache

#### Scenario: Index component fetching with fallback
- **WHEN** upstream answerbook index service fails or is unreachable
- **THEN** system falls back to the latest valid local cached index constituent list

### Requirement: Batch Fundamental Data Retrieval
The system SHALL batch or concurrently retrieve fundamental valuation metrics (PE, Forward PE, PB, EV/EBITDA) with rate limiting and error resiliency.

#### Scenario: Successful fundamental data extraction
- **WHEN** fundamental metrics are requested for a valid ticker
- **THEN** system returns sanitized dictionary containing PE, PB, Forward PE, and EV/EBITDA

#### Scenario: Missing or failing fundamental data
- **WHEN** ticker has incomplete fundamental data or provider raises an error
- **THEN** system returns None or NaN for missing fields without interrupting pipeline execution
