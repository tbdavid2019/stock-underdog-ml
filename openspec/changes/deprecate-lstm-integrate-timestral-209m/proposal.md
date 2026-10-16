# Proposal: Deprecate LSTM & Integrate Mistral Timestral-209M for Dual Foundation Model Resonance

## Why

The current prediction pipeline relies on an outdated single-layer LSTM model developed in 1997. In volatile financial time-series, this recurrent architecture suffers from severe vanishing gradients and regression towards the mean, collapsing into the random walk trap ($\hat{y}_{t+1} \approx y_t$, predicting yesterday's price). Even worse, the system's flagship signal—**多維共振 (Multi-Dimensional Resonance)**—was historically hardcoded to inherit the LSTM's target price and potential percentage directly, degrading the quality of resonance recommendations despite Google TimesFM 2.5 200M's superior accuracy.

Mistral AI recently released **Timestral-209M** (`mistralai/Timestral-209M`), a state-of-the-art decoder-only foundation model pre-trained on diverse time-series data with a 32,768-step context window, ranking #1 on GIFT-Eval, TIME, and fev-bench. Host verification confirms our production machine (`10.9.0.99`, 16-core Intel i7-12700, 40GB free RAM) can run CPU batch inference in 50–100ms. By replacing the archaic LSTM with Timestral-209M and fusing it with TimesFM 2.5, we elevate the system into an institutional-grade **Dual Foundation Model Quantitative Trading Engine** (Google TimesFM × Mistral Timestral ∩ Institutional Flow ∩ XuanTie Setup).

## What Changes

- **BREAKING: Complete Retirement of LSTM Architecture**:
  - Deprecate and remove `models/lstm.py` and its training/inference routines from the daily active pipeline.
  - Remove LSTM-specific database columns (`lstm_predicted_price`, `lstm_potential`) from primary write routines while preserving schema backward compatibility.
  - Deprecate `/api/v1/predictions/lstm/*` REST endpoints in favor of foundation model routes (retaining legacy aliases for compatibility).
- **Integration of Mistral Timestral-209M Foundation Model**:
  - Implement `models/timestral_model.py` providing `TimestralModel` and `TimestralStrategy` conforming to `BaseStrategy`.
  - Perform probabilistic multi-quantile forecasting (P10, P50, P90) across a multi-step horizon (1 to 5 trading days).
  - Extract median target price (P50), expected upside potential, and asymmetric risk-reward ratio (`(P90 - Current) / (Current - P10)`).
- **Overhaul of 多維共振 (Dual Foundation Model Resonance)**:
  - Redefine Resonance qualification criteria: true resonance now requires **Dual-FM Bullish Consensus** (both Google TimesFM and Mistral Timestral projecting positive forward return) ∩ **XuanTie MA Trend & Pullback** ∩ **Institutional Smart Money Accumulation**.
  - Synthesize Resonance target price from an ensemble of TimesFM target price, Timestral P50 median target, and technical support/resistance levels, eliminating the dependency on single-model outputs.
- **Backend Pipeline & Storage Modernization**:
  - Update `pipeline/orchestrator.py` to coordinate Stage 2 ML inference with TimesFM and Timestral concurrently or sequentially with memory caching.
  - Update `data/duckdb_manager.py` schema to store `timestral_predicted_price`, `timestral_potential`, `timestral_p10`, `timestral_p90`, and synthesized dual-FM metrics.
- **Multi-Channel Notification & Narrative Updates**:
  - Update `evaluators/formatter.py`, `notifier_dual.py`, and `evaluators/ai_narrative.py` to format dual-model outputs (`🤖 TimesFM` and `🔮 Timestral`), displaying quantile confidence intervals and dual-model consensus badges in Telegram, Discord, and Email alerts.
- **Financial Dashboard & Charting Overhaul**:
  - Update `api/templates/partials/tab_history.html` and `api/static/js/chart-manager.js` to render the Timestral target trajectory with P10–P90 confidence bands alongside historical closing prices and TimesFM forecasts.
  - Update `api/static/js/app.js` strategy dropdowns and table renderers to replace LSTM badges with Timestral badges.

## Capabilities

### New Capabilities
*(None; all changes modify or upgrade existing quant capabilities.)*

### Modified Capabilities
- `stock-strategy-engine`: Replace requirement for LSTM model strategy with Mistral Timestral-209M foundation model strategy, outputting multi-quantile forecast distributions (P10/P50/P90).
- `composite-evaluation`: Redefine Triple/Multi-Dimensional Resonance from LSTM-reliance to Dual Foundation Model Consensus (Google TimesFM 2.5 200M × Mistral Timestral 209M) combined with XuanTie technical setup and institutional flow.
- `fastapi-rest-service`: Update prediction query schemas and REST endpoints to return Timestral predictions and Dual-FM consensus metrics, providing backward-compatible redirects for legacy `/lstm/*` paths.
- `web-dashboard-ui`: Update dashboard filters, stock history charts, and recommendation tables to visualize Mistral Timestral forecasts, confidence envelopes, and Dual-FM consensus badges.

## Impact

- **Model Execution & Dependencies**: Adds `timestral` dependency (or standalone HuggingFace Transformers wrapper with `mistralai/Timestral-209M`). PyTorch CPU inference on `10.9.0.99` runs under ~100ms per ticker.
- **DuckDB Schema**: Adds Timestral columns (`timestral_predicted_price`, `timestral_potential`, `timestral_p10`, `timestral_p90`) to `predictions` table. Existing queries are preserved.
- **API Contracts**: REST API responses from `/api/v1/predictions/latest`, `/resonance`, and `/history/{ticker}` add `timestral_*` fields.
- **Frontend Dashboard**: Vue 3 / Alpine / Vanilla JS state in `app.js` and Chart.js datasets in `chart-manager.js` display Timestral and Dual Resonance instead of LSTM.
- **Notification Services**: Telegram, Discord, and Email report templates display both TimesFM and Timestral scores and consensus tags.
