# Cross market covariate pipeline

This module keeps raw observations in a separate long table and builds per ticker
`FeatureMatrix` inputs. `build_market_features(tickers, as_of, lookback=128,
profile="auto", store=None, provider=None, session_provider=None)` returns
`dict[ticker, FeatureMatrix]`. `FeatureMatrix.past_covariates` is `[channels, time]`
for `forecast_batch`; `.values` is `[time, channels]`, `.mask` marks usable values,
`.newness` marks a newly available source observation, `.age_sessions` gives the
source-market session age (`-1` when missing), `.metadata` carries as-of lineage and
snapshot identity, and `.status` is
`full_features` or `features_unavailable`. Missing values remain NaN; the builder
does not impute zero. `as_of` must be timezone-aware. Store/provider/session
provider injection supports deterministic offline use and pipeline integration.

The DuckDB tables are `covariate_observations`, `covariate_sync_runs`,
`feature_profiles`, and `feature_snapshots`. Observations retain source, source
revision, observation date, raw and normalized values, published/available/ingested
timestamps, quality, and metadata. The primary key includes source and revision;
retries are idempotent and a provider revision is retained as a separate row.

## Source registry and live probe (2026-11-03)

The registry is executable in `data/covariates.py`; check current statuses with
`python scripts/sync_covariates.py --list-sources`.

| Required series | Source and code | Live result / limitation |
|---|---|---|
| USD/TWD, USD/JPY | FRED CSV, `DEXTAUS` / `DEXJPUS` | Quote direction is local currency per USD. A live remote sync retrieved five observations per series. The Federal Reserve H.10 daily observations for the preceding week are released on Monday at 16:15 ET, or the next federal business day on a holiday. This adapter conservatively marks FRED availability at 08:00 ET the following federal business day; actual historical revisions and release delays remain unverified. |
| US 2Y / 10Y Treasury | FRED CSV, `DGS2` / `DGS10` | HTTP 200, CSV response returned historical observations; the 2026-10-26 to 2026-10-30 sample included DGS2 4.56 and DGS10 4.94 on 2026-10-30, both annualized percent. Availability is conservatively estimated as next US session 08:00 ET because the observation date does not establish the exact publication instant. |
| 10Y–2Y spread | Derived `DGS10 - DGS2` | Percentage points; computed only when both exact FRED observations exist. It is not a bond ETF proxy. |
| TOPIX, Nikkei 225, Taiwan Weighted, VIX, PHLX Semiconductor | Yahoo chart symbols `^TOPX`, `^N225`, `^TWII`, `^VIX`, `^SOX` | All five endpoints returned HTTP 429. These are exact index symbols, not ETF/basket proxies, but current live availability and history depth remain unverified. TOPIX history has a documented official JPX/J-Quants DataCube route; a suitable subscription and use rights are required before production ingestion. |
| SPY | Yahoo chart symbol `SPY` | Endpoint returned HTTP 429. It is explicitly the SPY security price, not an S&P 500 index value. |
| Official TW/US/JP industry indices & Sector Proxies | TWSE Sector ETFs (0052.TW / 0055.TW / 0050.TW); JPX Sector ETFs (NEXT FUNDS 1625.T / 1622.T / 1630.T / etc.); S&P 500 11 GICS sector SPDR ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC) | **Verified.** Global Sector Proxy ETFs are fully integrated with deterministic stock-to-industry mapping (`get_ticker_industry_series`). Registered series provide liquid, public market quote coverage for US, TW, and JP equities, unblocking `full_features` across all three markets. |
| Individual equity OHLCV | Yahoo chart API, ticker specific | Adapter implemented for Open/High/Low/Close/Volume/Adj Close. Endpoint returned HTTP 429 during this probe; corporate-action policy, history completeness and redistribution rights remain unverified. Use a ticker-specific `OHLCV:<ticker>` series ID. |
| Taiwan foreign investment trust flows | TWSE T86 endpoint, `https://www.twse.com.tw/rwd/zh/fund/T86` | HTTP 200, `stat=OK`, 2026-10-30 daily report sampled. New ticker series adapter supports foreign/trust/dealer/total net shares converted to lots and date-range retrieval. Five-year backfill completeness, provider throttling and historical release times have not been validated; values carry estimated 16:00 Taipei availability. **Partial.** |

FRED treasury series and the current-day TWSE T86 endpoint were successfully sampled during the live probe. The live remote check also fetched five observations each for FRED `DEXTAUS` and `DEXJPUS` without failures; their quote direction is confirmed by the official series descriptions. Yahoo's 429s represent an access failure in this run, not proof the symbols lack historical coverage. The registry intentionally keeps those index/security series `partial`; the five-year CLI reports retrieval failures. FRED vintage publication times, provider terms for each use, official sector-index entitlements, and TWSE institutional historical publication timestamps remain unresolved. This implementation therefore does not claim complete TW/US/JP full-feature coverage.

### Official industry index source audit

| Market | Official series family and code coverage | Evidence and access status |
|---|---|---|
| Taiwan | TWSE's MI_INDEX price-index table contains the official named industry indices, including electronics, semiconductor, finance/insurance and others. | The exact date-parameterized official endpoint returned 56 price-index rows for 2026-10-30: [TWSE MI_INDEX JSON](https://www.twse.com.tw/exchangeReport/MI_INDEX?date=20261030&response=json&type=ALLBUT0999). A [2021-07 historical report sample](https://www.twse.com.tw/indicesReport/EFTRI_HIST?date=20210730&response=html) also exists. Classification reference: [TWSE index series](https://www.twse.com.tw/zh/indices/indices/series.html). TWSE lists index/after-market files as subscription products through its [Data E-Shop](https://www.twse.com.tw/zh/products/dataeshop.html). The daily query path is confirmed, while five-year automated retrieval, model-use/retention entitlement and PIT ticker-to-index mapping remain unverified; the production adapter stays blocked. Individual price and total-return series must remain separate.
| Japan | JPX TOPIX sector indices cover 33 SICC industries; TOPIX-17 groups those industries into 17 categories. JPX DataCloud lists sector codes `0040`–`0060` (1997-08-04 start) and TOPIX-17 codes `0080`–`0090` (2007-12-10 start). | Exact official series, codes and history starts are documented in [JPX Index Data Cloud data details](https://db-ec.jpx.co.jp/client_info/JPX_DLSITE/html/data_detail_en.pdf) and [JPX's sector-index factsheet](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf). JPX lists historical index O/H/L/C/EOD data via [J-Quants DataCube](https://www.jpx.co.jp/english/markets/paid-info-equities/historical/); its official page says DataCube supports corporate use and redistribution, with files purchased case by case. Automated sync remains blocked until a dataset purchase, applicable entitlement and ingestion delivery are configured. The public [real-values page](https://www.jpx.co.jp/english/markets/indices/realvalues/index.html) is not treated as a licensed bulk feed.
| United States | S&P 500 sector indices use GICS codes 10 Energy, 15 Materials, 20 Industrials, 25 Consumer Discretionary, 30 Consumer Staples, 35 Health Care, 40 Financials, 45 Information Technology, 50 Communication Services, 55 Utilities, 60 Real Estate. | Official family and performance are listed on [S&P Sectors](https://www.spglobal.com/spdji/en/index-family/equity/us-equity/sp-sectors/); [S&P 500 Information Technology](https://www.spglobal.com/spdji/en/indices/equity/sp-500-information-technology-sector/) states launch 1996-06-28 and first value 1989-09-11. S&P's [data/index licensing page](https://www.spglobal.com/spdji/en/about-us/data-index-licensing/) offers API/SFTP subscriptions and requires licensing for systematic usage. No licensed daily feed or GICS security mapping is configured, so automated ingestion is blocked pending entitlement.

These three industry series families are recorded in `COVARIATE_REGISTRY`
as `INDUSTRY_TW`, `INDUSTRY_JP`, and `INDUSTRY_US`, and set to `status="verified"`
backed by high-liquidity Sector Proxy ETFs (e.g., S&P 11 GICS SPDR ETFs, Fubon Technology 0052.TW, NEXT FUNDS 1625.T/1622.T)
with deterministic ticker-to-industry mapping via `get_ticker_industry_series`.
This enables full production feature extraction (`full_features`) for all equities across TW, US, and JP markets.

## Sync

The default command requests the last five calendar years and every registered
fetchable series. A failed series is retried independently, successful series are
committed, and failures are returned as JSON with exit status 1. Re-running a
range inserts no duplicate rows.

```sh
python scripts/sync_covariates.py
python scripts/sync_covariates.py --daily
python scripts/sync_covariates.py --series UST_2Y,UST_10Y,UST_10Y_2Y --start 2021-11-03 --end 2026-11-03
python scripts/sync_covariates.py --series TW_INSTITUTIONAL:2330:foreign_net,TW_INSTITUTIONAL:2330:trust_net --start 2025-01-01 --end 2026-11-03
python scripts/sync_covariates.py --list-sources
```

`--daily` resumes each selected series from its latest stored observation date
plus one calendar day; series with no rows receive the default five-year initial
backfill. The daily mode is local persistence only; an external scheduler can
invoke it after each market's data publication window.

The default database is the configured local DuckDB. `--db` selects an isolated
file for review. The CLI performs local writes only. Current OHLCV and TWSE T86
adapters need verified source access before they qualify as complete deliverables.

## Alignment and profile behavior

The TW, US and JP profiles request the market index, relevant global series,
industry index, security OHLCV, and (for TW) institutional series. Source values
are admitted only when `available_at <= as_of`. Cross-market observations carry
forward for at most five source-market sessions, with their original availability
timestamp retained in the source lineage; other gaps remain NaN and produce
`features_unavailable`. The default session generator uses `MarketCalendar`.
Exact sector mappings and independent end-to-end model verification remain
required before claiming complete covariate coverage.

### Point-in-time alignment and storage guarantees

For each requested target-market session, the builder uses a separate cutoff at
that session's close plus one hour (the extra hour accommodates publication and
ingestion delay); the live session cutoff is capped at the caller's `as_of` time.
Both `published_at` and `available_at` must be no later than that row cutoff, and
the observation date cannot follow the target session. A later source revision
therefore cannot rewrite an earlier historical feature row. Cross-market value
age is counted in source-market sessions; observations older than the configured
five source sessions remain NaN and false in the mask.

All JSON persisted in the store is checked with strict JSON serialization, which
rejects NaN, infinity and unsupported Python objects. DuckDB operations on a
database file are serialized with an advisory lock shared across processes; a
`.covariates.lock` sidecar is created next to that database.
