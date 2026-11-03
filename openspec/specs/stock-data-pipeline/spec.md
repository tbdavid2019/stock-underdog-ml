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

### Requirement: Required cross market covariate coverage
系統 SHALL 在本次變更交付美元兌台幣、美元兌日圓、美債2年及10年殖利率與10年減2年利差、TOPIX、日經225、台灣加權、SPY、VIX、費半、台美日產業指數、個股價量及台股外資／投信資料。每項須有可運作來源與具版本的定義、單位、歷史覆蓋及品質狀態，僅列候選來源不構成完成。

#### Scenario: Acceptance of the complete data inventory
- **WHEN** 驗收本次協變量資料交付
- **THEN** 每項均可查詢歷史範圍與最新有效值，美元兌台幣、美債殖利率、TOPIX及各市場產業指數亦列為必要驗收項目

### Requirement: Historical backfill and daily persistence
系統 SHALL 為必要協變量提供可設定期間的歷史回填、每日增量更新與持久化，保存來源、觀測日、公布／可得時間、抓取時間及修訂版本；重跑須冪等，來源失敗保留最後有效資料並標示stale與失敗原因。

#### Scenario: Restarting an interrupted import
- **WHEN** 歷史回填中斷後重跑或同日更新執行兩次
- **THEN** 紀錄保持唯一且可續傳，價格或利率修訂另存版本，失敗資料不覆蓋有效值

### Requirement: Market specific prediction feature profiles
系統 SHALL 依標的市場套用協變量組合。台股包含加權、產業、價量、法人及美元兌台幣；美股包含SPY、產業、價量、VIX與美債殖利率及利差；日股包含日經、TOPIX、產業、價量及美元兌日圓。國際費半／VIX等跨市場資料須依版本化設定及當時可得性使用。

#### Scenario: Predictions for Taiwan, United States and Japan
- **WHEN** 管線分別預測三個市場的股票
- **THEN** 各模型收到對應市場特徵、來源及可用性遮罩，執行紀錄可證明其實際輸入；資料僅用於事後風控不足以通過此驗收

### Requirement: Point in time feature alignment
系統 SHALL 以決策時間對齊協變量，只使用當時已公布及可取得的完整觀測；休市、時差、發布延遲及修訂須保留狀態，前值沿用須受明示新鮮度上限限制，禁止未來值補洞或將缺值當零。

#### Scenario: Taiwan forecast before United States session closes
- **WHEN** 台股預測的決策時點早於某筆美股或美債資料可得時間
- **THEN** 採上一筆當時可得且仍合新鮮度的資料，否則標缺失並輸出降級原因

### Requirement: Explicit financial series semantics
系統 SHALL 保存匯率報價方向與單位、殖利率年期及百分比尺度、指數價格／報酬類型及產業映射版本；ETF或自建板塊代理須明示與原指數的差異，不得默認等同。缺乏所需產業映射須揭露覆蓋缺口。

#### Scenario: Provider changes symbols or units
- **WHEN** 來源切換、匯率報價反轉或殖利率尺度變更
- **THEN** 驗證正規化後數值與原定義一致，設定及資料版本更新且原始值仍可追溯
