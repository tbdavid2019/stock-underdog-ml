# stock-strategy-engine Specification

## Purpose
Standardizes the strategy interface and provides a pluggable strategy registry for technical indicators, rule-based filters, and machine learning prediction models.

## Requirements

### Requirement: Uniform Strategy Interface
The system SHALL require all stock selection and prediction strategies to inherit from `BaseStrategy` and implement a standard evaluation interface that returns a structured `StrategyResult` object.

#### Scenario: Strategy execution producing standard result
- **WHEN** a strategy evaluates a stock's historical price and fundamental context
- **THEN** it returns a `StrategyResult` containing ticker, strategy_name, hit status, score, signals, and metric attributes

#### Scenario: Insufficient data handling
- **WHEN** input price data has fewer bars than required by the strategy's declared lookback
- **THEN** strategy returns an un-hit result with a reason rather than raising an unhandled exception

### Requirement: Dynamic Strategy Registry and Discovery
The system SHALL provide a centralized `StrategyRegistry` allowing arbitrary new strategies to be registered via decorators or class discovery and enabled via configuration without modifying the pipeline runner.

#### Scenario: Enabling registered strategies via configuration
- **WHEN** user configures `ENABLED_STRATEGIES=["xuantie", "lstm", "custom_momentum"]`
- **THEN** pipeline dynamically instantiates and executes only the configured strategies

#### Scenario: Strategy feature requirements declaration
- **WHEN** a strategy specifies required lookback periods or indicators
- **THEN** data pipeline guarantees the necessary data is preloaded before strategy execution

### Requirement: XuanTie Swing Strategy Implementation
The system SHALL evaluate the XuanTie moving average strategy (major trend filter via MA60 slope + minor pullback filter via MA60/MA120 tolerance).

#### Scenario: XuanTie buy point matched
- **WHEN** MA60 is sloped upwards over the lookback window and price is within the tolerance band of MA60 or MA120
- **THEN** XuanTie strategy flags is_hit as True and specifies the pullback type

### Requirement: LSTM Model Strategy Implementation
The system SHALL train or infer next-day closing price for a stock using LSTM architecture and output predicted potential percentage.

#### Scenario: LSTM prediction calculated
- **WHEN** valid historical sequence is provided to trained LSTM model
- **THEN** model outputs predicted next-day price and calculates expected potential percentage
