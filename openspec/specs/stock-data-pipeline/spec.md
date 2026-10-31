# stock-data-pipeline Specification

## Purpose
Manages stock index components, historical market OHLCV price series, and fundamental metrics retrieval with multi-tiered caching and batch operations.

## Requirements

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

### Requirement: Dynamic Multi-Market Universe Expansion & Delisting Filtering
The system SHALL retrieve expanded stock universes for Taiwan (1,000 stocks) and US (500+ stocks) with automatic delisting and IPO lifecycle filtering.

#### Scenario: Dynamic Taiwan 1000 constituent selection
- **WHEN** Taiwan universe is requested via `get_tw_universe(limit=1000)` or `get_tw1000_stocks()`
- **THEN** system queries DuckDB `tw_daily_bars` and `market_universe` to return the top 1,000 active equities ranked by liquidity/turnover, ensuring TW0050, TW0051, and active industry supply chain nodes are included.

#### Scenario: Delisted or suspended stock pruning
- **WHEN** a stock has delisted or ceased trading on TWSE/TPEx or US exchanges
- **THEN** system filters out tickers with non-ACTIVE listing status or zero trading volume, preventing obsolete tickers from entering model inference.

#### Scenario: Full US S&P 500 retrieval
- **WHEN** `get_sp500_stocks()` is invoked without explicit limit
- **THEN** system retrieves all 503 constituents plus strategic AI/space high-growth leaders (~510 total tickers).

### Requirement: Chunked Batch Market Data Fetching
The system SHALL chunk batch downloads into safe block sizes (e.g. 200 tickers per sub-batch) to prevent network socket drops or timeouts during large-scale universe prefetching.

#### Scenario: Chunked prefetching for large universes
- **WHEN** `get_stock_data_batch` is called with more than 200 tickers
- **THEN** system processes the tickers in chunks of up to 200, combining the results into a single consolidated dictionary.
