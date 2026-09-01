# macro-regime-filter Specification

## Purpose
Monitors US benchmark indices (SPY, VIX, SOX) to evaluate global macro risks, determine market regime, and dynamically regulate portfolio exposure levels.

## Requirements

### Requirement: US Macro Index Ingestion and Regime Classification
The system SHALL fetch historical and real-time closing data for US benchmark tickers (`^GSPC`/`SPY`, `^VIX`, `^SOX`) to compute macro risk state and exposure percentage (0.0 to 1.0).

#### Scenario: Normal Bull Market Condition
- **WHEN** SPY is above its 60-day moving average and VIX is under 22
- **THEN** system classifies the regime as "Bullish" with 100% recommended exposure (1.0)

#### Scenario: Moderate Risk Condition
- **WHEN** SPY is below its 60-day moving average and VIX is between 22 and 28
- **THEN** system classifies the regime as "Cautious/Defensive" and reduces exposure to 20% - 40%

#### Scenario: Extreme Panic VIX Shutdown
- **WHEN** VIX exceeds 28.0
- **THEN** system flags an "Extreme Panic Warning" and sets market exposure to 0.0 (full cash / halt new buying)

#### Scenario: Semiconductor Sector SOX Gate
- **WHEN** SOX is below its 60-day moving average with negative short-term momentum
- **THEN** system mandates half-position exposure cap (50%) for tech/semiconductor-related stocks
