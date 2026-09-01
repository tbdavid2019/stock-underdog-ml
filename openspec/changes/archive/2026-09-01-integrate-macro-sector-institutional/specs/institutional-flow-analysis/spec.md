## Purpose

Ingests official TWSE/TPEX institutional trading and shareholding data to compute foreign investor, investment trust, and dealer accumulation metrics for Taiwan market equities.

## ADDED Requirements

### Requirement: Institutional Daily Buy/Sell Ingestion
The system SHALL fetch daily institutional net buy/sell volumes (TWSE T86 and TPEX 3itrade reports) and compute rolling 5-day and 20-day cumulative net buy/sell quantities for Foreign Investors, Investment Trusts (投信), and Dealers.

#### Scenario: Investment Trust Buying Streak Detection
- **WHEN** Investment Trust (投信) registers consecutive net buying for 3 or more days or significant 5-day accumulation
- **THEN** system flags the stock with a positive institutional accumulation score and `投信連買` tag

#### Scenario: Foreign and Trust Synchronization (土洋合做)
- **WHEN** both Foreign Investors and Investment Trusts have net positive accumulation over a 5-day window
- **THEN** system flags the stock with a high-conviction `土洋合買` badge

### Requirement: Foreign Shareholding Ratio Shift Analysis
The system SHALL ingest official foreign shareholding percentages (TWSE MI_QFIIS) to compute 20-day and 60-day foreign ownership ratio delta.

#### Scenario: Foreign Holding Expansion
- **WHEN** foreign shareholding percentage increases over the 20-day observation window
- **THEN** system attributes positive institutional momentum score to the stock
