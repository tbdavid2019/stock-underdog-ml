# Proposal: Scale Daily Quant Forecasting Universe to 1,500 Stocks (1,000 TW + 500 US)

## Problem Statement
The current daily quantitative forecasting pipeline only covers ~148 Taiwan stocks (TW0050 50 large caps + TW0051 100 mid caps with overlap filtering) and restricts the US market to 110 stocks due to a historical `limit=110` default parameter.
However:
1. Production benchmarking on `10.9.0.99` (16 CPU cores, 53GB RAM) demonstrates that zero-shot foundation models (TimesFM-200M and Timestral-209M) evaluate 100 series in **~2.5 seconds** combined, allowing 1,000 series to complete vectorized inference in under **30 seconds**.
2. Taiwan's primary AI high-growth beneficiaries and supply-chain champions (e.g., 弘塑, 雙鴻, 富世達, 上詮, 光聖, 萬潤, 均華, 旺矽, 健策) are concentrated in OTC (TPEx 櫃買) and mid/small caps outside TW0050 and TW0051.
3. The US market upstream source (`answerbook.david888.com/SP500`) provides all 503 S&P 500 constituents, and high-beta AI/space tech champions (SPCX, DXYZ, ASTS, RKLB, CRWV, FN, CLS) should be included without artificial limits.
4. The stock universe must dynamically adapt to market shifts: automatically detecting and including new IPOs (新上市/櫃掛牌) on day 1 while pruning delisted (下市/下櫃) or zero-volume zombie tickers.

## Proposed Solution
1. **Dynamic Taiwan 1,000 Stock Universe (`get_tw_universe` / `get_tw1000_stocks`)**:
   - Construct a tiered active universe combining:
     - Core Tier: TW0050 (50), TW0051 (100), TPEx50 (50), and AI supply chain graph nodes (`tw_industry_nodes`, 88 nodes).
     - Liquidity Tier: Dynamically ranked by 20-day liquidity/turnover (`trade_value`) from DuckDB `tw_daily_bars` & `market_universe` up to 1,000 unique tickers.
     - Delisting & IPO Guard: Filter strictly for `listing_status = 'ACTIVE'`, `trade_value > 0`, and trading activity in the latest snapshot. Delisted stocks cease receiving bars and are automatically purged; IPOs with active initial volume are immediately included.
2. **Full S&P 500 + Strategic US 500 Universe (`get_sp500_stocks`)**:
   - Remove the hardcoded `limit=110` default; support the full S&P 500 constituent list (503 stocks) plus high-conviction space/AI tech leaders (~510 stocks total).
   - Inactive / delisted ticker resilient handling: yfinance batch fetch skips non-existent tickers cleanly.
3. **Pipeline Segmentation & Chunked Prefetching**:
   - Implement batch chunking (e.g. chunks of 200 tickers) in `StockFetcher.get_stock_data_batch` to prevent yfinance socket exhaustion or rate throttling when fetching 1,000 stocks.
   - For `--market tw`: partition execution into `台灣50` (50), `台灣中型100` (100), and `台股活躍精選` (850) to maintain backward compatibility, optimize Telegram notification chunking, and prevent metric dilution, while supporting `--index 台股1000` for monolithic analysis.
   - For `--market us`: execute the unified `SP500` pool (500+ stocks).
4. **Scheduled Automated Maintenance**:
   - Confirm crontab executions for `scripts/sync_twse_market.py` and `scripts/sync_global_universe.py` precede every daily morning and evening quant run.
