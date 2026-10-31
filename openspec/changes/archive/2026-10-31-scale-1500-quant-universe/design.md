# Design: Dynamic 1,500 Stock Quantitative Universe Architecture

## Architectural Overview
```
┌────────────────────────────────────────────────────────────────────────┐
│                   Global Security Directories                          │
│   TWSE/TPEx OpenAPI (1,945)   │   Nasdaq Trader & Answerbook (13,204)  │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
       scripts/sync_twse_market.py       scripts/sync_global_universe.py
                    │                                │
                    ▼                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 DuckDB Global Store (stock_quant.duckdb)               │
│   • market_universe (1,945 TW + 13,204 US active symbols)              │
│   • tw_daily_bars (daily OHLCV, volume, trade_value)                   │
│   • tw_industry_nodes (88 supply chain nodes)                          │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
       Dynamic TW1000 Selector             US500 Selector
       - TW0050 (50)                       - S&P 500 (503)
       - TW0051 (100)                      - Nasdaq 100 AI pure-plays
       - TPEx50 (50)                       - Space / High-Beta Tech
       - Supply Chain Nodes (88)           (ASTS, RKLB, SPCX, CRWV...)
       - Top liquid active equities (850)
                    │                                │
                    ▼                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      StockFetcher Prefetch Engine                      │
│   • Chunked Batch Download (chunks of 200 tickers)                     │
│   • Multi-tier fallback: Cache -> yfinance -> DuckDB tw_daily_bars     │
│   • Zero-volume / delisted auto-pruning                                │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Foundation ML Strategy Engine                     │
│   • TimesFM 200M (Zero-shot, vectorized batch inference)               │
│   • Timestral 209M (P10/P50/P90 quantile probabilistic inference)      │
│   • XuanTie rule-based indicator engine                                │
└────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Dynamic Taiwan Universe Selector (`StockFetcher.get_tw_universe`)
- **Query Logic**:
  ```sql
  SELECT u.normalized_symbol, u.local_symbol, u.name_local, SUM(b.trade_value) as total_turnover
  FROM market_universe u
  JOIN tw_daily_bars b ON u.normalized_symbol = b.ticker
  WHERE u.market = 'TW'
    AND u.listing_status = 'ACTIVE'
    AND u.security_type = 'EQUITY'
    AND b.date >= (SELECT date(MAX(date), '-20 day') FROM tw_daily_bars)
    AND b.trade_value > 0
  GROUP BY u.normalized_symbol, u.local_symbol, u.name_local
  ORDER BY total_turnover DESC
  ```
- **Guaranteed Inclusions**:
  1. `TW0050` (50 stocks)
  2. `TW0051` (100 stocks)
  3. `tw_industry_nodes` (88 nodes from knowledge graph)
  4. Top turnover stocks from TWSE and TPEx up to `limit=1000`.
- **Dynamic IPO / Delisting Adaptation**:
  - New IPOs: Added on Day 1 by `sync_twse_market.py` OpenAPI fetch. High trading volume immediately ranks them into the pool.
  - Delisted / Suspended stocks: Once delisted, TWSE/TPEx omits them; `tw_daily_bars` receives 0 trades. The filter `b.trade_value > 0` and `listing_status = 'ACTIVE'` automatically purges them.

### 2. Full US 500+ Universe Selector (`StockFetcher.get_sp500_stocks`)
- Update signature to `get_sp500_stocks(cls, limit: Optional[int] = None) -> List[str]`.
- Default to fetching all 503 constituents from `https://answerbook.david888.com/SP500`.
- Append AI infrastructure and space technology leaders (`SPCX`, `DXYZ`, `ASTS`, `RKLB`, `CRWV`, `FN`, `CLS`).

### 3. Pipeline Chunking & Execution Partitioning
- In `StockFetcher.get_stock_data_batch`:
  - When ticker count exceeds 200, divide into chunks of 200 to prevent HTTP connection drops or yfinance thread pool lockups.
- In `PipelineOrchestrator.run_all_indices`:
  - For `market="tw"`:
    - `"台灣50"`: 50 blue chips
    - `"台灣中型100"`: 100 mid caps
    - `"台股活躍精選"`: 850 high-turnover growth & OTC champions (disjoint set)
    - Total = 1,000 distinct stocks.
  - For `market="us"`:
    - `"SP500"`: 500+ US stocks.
  - For `--index 台股1000`:
    - Direct monolithic 1,000-stock execution.

### 4. API & UI Integration
- Update Web Dashboard market dropdown to include `台股活躍精選 (TW-Active)` and `台股全市場1000 (TW1000)`.
- Support API query `GET /api/v1/predictions/resonance?index_name=台股活躍精選`.
