## Why

The current stock selection pipeline in `stock-underdog-ml` relies primarily on single-stock technical indicators (XuanTie MA60/120 pullbacks), ML LSTM price predictions, and basic valuation metrics (PE/PB). However, empirical quant research from high-performing systems (such as `tw_stocker` and `tw-institutional-stocker`) demonstrates that individual stock performance in Taiwan and US markets is heavily dictated by three higher-order dimensions:
1. **US Macro Risk Regimes (`SPY`, `VIX`, `SOX`)**: 台股科技股與大盤高度連動美股，在市場極端恐慌 ($VIX > 28$) 或費半破季線時，個股技術買點勝率大幅下降，需要頂層宏觀風控門檻。
2. **Sector Capital Rotation (板塊資金輪動)**: 資金往往在電子、半導體、金融、傳產等 7 大板塊中輪動，順應強勢板塊可顯著提高 Alpha。
3. **Institutional Flow (三大法人籌碼面)**: 投信連續買超與外資/投信同步買超（土洋合做）是台股中短期爆發力的關鍵推手。

Integrating these components into our modular strategy engine and composite evaluator will significantly improve signal robustness, filter out false breakdowns, and enable "Triple-Resonance" (三重共振) high-conviction recommendations.

## What Changes

- **Macro Regime Gate (`macro-regime-filter`)**: Fetch `^GSPC` (SPY), `^VIX`, and `^SOX` pre-flight to compute dynamic market exposure (0% ~ 100%) and issue macro safety warnings.
- **Sector Rotation Engine (`sector-rotation-strategy`)**: Track 10/15/20-day momentum across 7 core market sectors and select top-performing sectors.
- **Institutional Flow Engine (`institutional-flow-analysis`)**: Ingest TWSE/TPEX institutional buy/sell data (T86/MI_QFIIS) to compute 5D/20D net institutional accumulation, foreign shareholding ratio shifts, and investment trust buy streaks.
- **Strategy Registry Expansion (`stock-strategy-engine`)**: Register `SectorRotationStrategy` and `InstitutionalFlowStrategy` into the standard strategy registry.
- **Composite Evaluator Upgrade (`composite-evaluation`)**: Combine Technical + LSTM + Institutional signals into "Triple Resonance" tags (`三重共振`), with macro exposure weighting.
- **Pipeline & Storage Enriched (`pipeline-orchestrator`)**: Include macro regime status and institutional accumulation in Telegram/Discord/Email notifications and Supabase database columns (`macro_regime`, `trust_net_5d`, `foreign_net_5d`).

## Capabilities

### New Capabilities
- `macro-regime-filter`: Monitors US benchmark indices (SPY, VIX, SOX) to classify macro market regime and dynamically set portfolio exposure levels (0.0 to 1.0).
- `sector-rotation-strategy`: Evaluates capital flow momentum across 7 industry sectors to identify and rank top-3 performing sectors and constituents.
- `institutional-flow-analysis`: Ingests and calculates Taiwan market institutional investor (Foreign, Investment Trust, Dealer) net accumulation and holding ratio momentum.

### Modified Capabilities
- `stock-strategy-engine`: Extends the strategy registry to support macro-conditioned and institutional-flow-based strategies.
- `composite-evaluation`: Adds multi-factor scoring rules for institutional momentum and triple-resonance qualification.
- `pipeline-orchestrator`: Incorporates macro regime pre-check in Stage 1 and dispatches enriched institutional metrics in notifications and persistence.

## Impact

- **Core/Pipeline**: `pipeline/orchestrator.py` adds pre-flight macro regime calculation.
- **Data Layer**: New data providers for TWSE/TPEX institutional flows and Yahoo Finance macro tickers (`^VIX`, `^SOX`, `SPY`).
- **Strategies**: New strategy classes in `strategies/sector_rotation.py` and `strategies/institutional.py`.
- **Evaluators**: Upgraded `evaluators/composite_evaluator.py` and formatters for multi-factor badge rendering.
- **Database & Alerts**: Supabase schema extension with non-breaking optional columns, and enriched Discord/Telegram notification cards.
