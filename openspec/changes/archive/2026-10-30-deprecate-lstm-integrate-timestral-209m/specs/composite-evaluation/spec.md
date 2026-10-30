# Spec Delta: composite-evaluation

## MODIFIED Requirements

### Requirement: Triple Resonance and Macro Adjusted Ranking
The system SHALL aggregate Technical (XuanTie), Dual Foundation Models (Google TimesFM and Mistral Timestral), and Institutional flow signals with Macro Regime exposure multipliers to compute final composite priority scores and identify Multi-Dimensional Resonance candidates.

#### Scenario: Triple Resonance Qualification
- **WHEN** a stock concurrently satisfies XuanTie technical pullback, positive Google TimesFM prediction, positive Mistral Timestral P50 potential, and positive institutional net accumulation
- **THEN** evaluator marks the stock with highest conviction priority and appends the `🏆多維共振` badge

#### Scenario: Resonance Target Price Synthesis
- **WHEN** a stock qualifies for Multi-Dimensional Resonance
- **THEN** evaluator synthesizes the resonance target price as a weighted ensemble of TimesFM target price, Timestral P50 forecast, and technical resistance, rather than relying on a single model

#### Scenario: Macro Exposure Discounting
- **WHEN** the US Macro Regime indicates Defensive or Panic status (exposure < 1.0)
- **THEN** evaluator applies exposure discount to final rank score and prepends macro risk warning badges to recommendations
