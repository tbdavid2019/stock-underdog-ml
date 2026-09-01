# sector-rotation-strategy Specification

## Purpose
Tracks capital flow and multi-period price momentum across core market sectors to identify top-performing industry sectors and recommend leading constituents.

## Requirements

### Requirement: Industry Sector Flow Momentum Tracking
The system SHALL map component stocks into core industry sectors (e.g. Semiconductor, Electronics, Financials, Traditional/Industrials, Shipping, Green Energy, Biotech) and compute rolling 10-day, 15-day, and 20-day sector-weighted momentum.

#### Scenario: Top Sector Identification
- **WHEN** daily sector flows and average price momentum are calculated
- **THEN** system ranks sectors and selects the top-3 performing sectors with positive average momentum

#### Scenario: Sector Risk Elimination
- **WHEN** a top-ranking sector's average 20-day return is below -3%
- **THEN** system disqualifies that sector from buy recommendations to avoid buying declining themes

### Requirement: Sector-Constituent Cross Ranking
The system SHALL evaluate constituents within the top-selected sectors using momentum and trend ranking to select top-performing candidates.

#### Scenario: Selecting Sector Leaders
- **WHEN** top-3 sectors are established
- **THEN** system ranks stocks within each sector by momentum (`close / close_20d`) and trend (`close > MA60`) to return top-3 leaders per sector
