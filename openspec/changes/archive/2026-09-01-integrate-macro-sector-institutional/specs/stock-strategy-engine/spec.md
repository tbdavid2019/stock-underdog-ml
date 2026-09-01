## ADDED Requirements

### Requirement: Macro and Sector Aware Strategy Implementations
The system SHALL provide `SectorRotationStrategy` and `InstitutionalFlowStrategy` conforming to `BaseStrategy` to evaluate sector momentum and institutional accumulation alongside technical and ML models.

#### Scenario: Executing Institutional Flow Strategy
- **WHEN** institutional trading data is supplied to `InstitutionalFlowStrategy`
- **THEN** it generates a `StrategyResult` containing institutional score, net volume metrics, and buy-streak signals

#### Scenario: Executing Sector Rotation Strategy
- **WHEN** multi-stock sector dataset is supplied to `SectorRotationStrategy`
- **THEN** it generates ranked candidate selections for stocks belonging to the leading sectors
