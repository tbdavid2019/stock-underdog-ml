# stock-data-pipeline Delta Specification

## ADDED Requirements

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
