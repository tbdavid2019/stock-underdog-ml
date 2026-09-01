## 1. US Macro Regime Gate Implementation

- [ ] 1.1 Create `data/macro.py` with `MacroRegimeAnalyzer` fetching `SPY`, `^VIX`, and `^SOX` to compute market regime and exposure level; verify with unit test `test/test_macro.py`.
- [ ] 1.2 Integrate macro pre-flight check into `pipeline/orchestrator.py` Stage 1; verify orchestrator attaches macro summary to analysis reports.

## 2. Taiwan Institutional Flow Data Ingestion

- [ ] 2.1 Implement `data/institutional.py` to ingest official TWSE (T86/MI_QFIIS) and TPEX daily institutional data with local disk caching; verify fetching TW50 tickers with unit test `test/test_institutional.py`.
- [ ] 2.2 Implement 5D/20D rolling accumulation calculation and investment trust (投信) consecutive buying streak detection; verify calculation accuracy with test assertions.

## 3. Sector Taxonomy and Sector Rotation Strategy

- [ ] 3.1 Define 7 core industry sectors mapping in `config.py` for TW50, TW100, and S&P500 constituents; verify sector coverage with test.
- [ ] 3.2 Implement `strategies/sector_rotation.py` conforming to `BaseStrategy` to compute rolling sector momentum (10D/15D/20D) and rank top-3 sectors; verify with unit test `test/test_sector_rotation.py`.

## 4. Institutional Strategy & Strategy Registry Expansion

- [ ] 4.1 Implement `strategies/institutional.py` conforming to `BaseStrategy` to generate institutional accumulation scores and signals; verify with unit test.
- [ ] 4.2 Register `SectorRotationStrategy` and `InstitutionalFlowStrategy` in `strategies/registry.py`; verify registry discovery and instantiation.

## 5. Composite Evaluator & Triple Resonance Upgrade

- [ ] 5.1 Update `evaluators/composite_evaluator.py` to aggregate technical, ML, and institutional signals with macro exposure multipliers; verify scoring logic with unit test `test/test_composite_evaluator.py`.
- [ ] 5.2 Add `三重共振` (Triple Resonance), `土洋合買`, `投信連買`, and `宏觀風控` badges to `evaluators/formatter.py`; verify console and notification report formatting.

## 6. Sinks Integration & End-to-End Verification

- [ ] 6.1 Update Supabase persistence in `database.py` to include optional columns (`macro_regime`, `trust_net_5d`, `foreign_net_5d`) with `json_safety.py` sanitization; verify database insert.
- [ ] 6.2 Execute full end-to-end multi-index analysis locally and verify complete report output, notifications formatting, and Supabase data integrity.
