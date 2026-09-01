## Context

Building upon the modular architecture established in `stock-underdog-ml` (see `proposal.md` and archived `2026-09-01-refactor-pipeline-architecture`), this design adds three complementary quantitative pillars: Macro Regime Risk Gates, Sector Flow Momentum, and Taiwan Institutional Flow Analytics.

## Goals / Non-Goals

**Goals:**
- Implement a standalone `MacroRegimeAnalyzer` fetching `SPY`, `^VIX`, and `^SOX` to compute portfolio exposure multipliers (0.0 to 1.0) and top-level market health flags.
- Ingest TWSE/TPEX official daily institutional trading data (`T86` and `MI_QFIIS`) to compute 5-day/20-day Investment Trust (投信) and Foreign Investor net buying trends.
- Implement a 7-sector momentum model classifying stocks and prioritizing top-3 capital-attracting sectors.
- Upgrade `CompositeEvaluator` to detect "Triple-Resonance" (XuanTie technical + LSTM bullish + Institutional accumulation) and discount scores when macro risk is elevated.
- Maintain 100% backward compatibility and zero downtime on daily cron execution.

**Non-Goals:**
- Ingesting granular Broker-Branch (券商分點) daily transaction tables due to high IP throttling risks on TWSE servers and excessive storage overhead.
- Replacing the existing concurrent Yahoo Finance data prefetcher.

## Decisions

### 1. Dedicated Institutional Data Provider with Cache & Fallback
- **Choice**: Implement `data/institutional.py` using official TWSE/TPEX REST endpoints (`https://www.twse.com.tw/rwd/zh/fund/T86`, `https://www.tpex.org.tw/web/stock/3invest/3itrade_hedge/3itrade_hedge_result.php`).
- **Rationale**: Direct official endpoints provide authoritative daily net buy/sell numbers without third-party API fees.
- **Alternative considered**: Scraping Goodinfo / Yahoo. Rejected due to anti-scraping CAPTCHAs and brittle HTML layouts.
- **Resiliency**: If TWSE API times out or fails (e.g. holidays / maintenance), the system gracefully logs a warning and proceeds with price + ML analysis without failing the pipeline.

### 2. Pre-Flight Macro Hook in Pipeline Orchestrator
- **Choice**: Orchestrator runs `MacroRegimeAnalyzer.evaluate_us_market()` at Stage 1 before processing individual stock indices.
- **Rationale**: Computing global macro regime once upfront avoids redundant downloads and allows all subsequent index runs (TW50, TW100, SP500) to reference the same macro safety state.
- **Rules**:
  - `VIX > 28.0` ➔ Exposure = 0.0 (Stop/Halt Alert)
  - `SPY < MA60` & `SOX < MA60` ➔ Exposure = 0.1 ~ 0.4 (Defensive Alert)
  - `SPY > MA60` & `VIX < 22.0` ➔ Exposure = 1.0 (Full Bullish)

### 3. Sector Classification & Momentum Scoring
- **Choice**: Maintain a standard sector taxonomy in `config.py` mapping TW50, TW100, and S&P500 tickers into 7 core sectors (Semiconductor, Electronics/AI, Financials, Industrials/Capex, Shipping/Logistics, Energy/Green, Healthcare/Biotech).
- **Ranking**: Weighted 10D (40%), 15D (30%), 20D (30%) price momentum across sector constituents. Disqualify sectors with negative average return (< -3%).

### 4. Triple Resonance Badge in Composite Evaluation
- **Badge Rules**:
  - `三重共振`: XuanTie is_hit + LSTM potential > +1.5% + Investment Trust 5D net buy > 0.
  - `土洋合買`: Foreign 5D net > 0 AND Trust 5D net > 0.
  - `投信連買`: Trust consecutive net buying >= 3 days.

## Risks / Trade-offs

- **[Risk: TWSE/TPEX Endpoint Rate Limiting or Schema Changes]** → Mitigation: Implement 2-second polite rate limiting, JSON schema validation, and persistent daily cache under `data/cache/institutional_*.json`.
- **[Risk: Computation Overhead on Multi-Stock Sector Ranking]** → Mitigation: Vectorized pandas matrix operations in Stage 2.
- **[Risk: Supabase Backward Compatibility]** → Mitigation: New columns in `predictions` (`macro_regime`, `trust_net_5d`, `foreign_net_5d`) are nullable, and payload is sanitized with `json_safety.py`.
