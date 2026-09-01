## ADDED Requirements

### Requirement: Triple Resonance and Macro Adjusted Ranking
The system SHALL aggregate Technical (XuanTie), ML (LSTM), and Institutional flow signals with Macro Regime exposure multipliers to compute final composite priority scores and identify Triple-Resonance candidates.

#### Scenario: Triple Resonance Qualification
- **WHEN** a stock concurrently satisfies XuanTie technical pullback, positive LSTM prediction, and positive institutional net accumulation
- **THEN** evaluator marks the stock with highest conviction priority and appends the `三重共振` badge

#### Scenario: Macro Exposure Discounting
- **WHEN** the US Macro Regime indicates Defensive or Panic status (exposure < 1.0)
- **THEN** evaluator applies exposure discount to final rank score and prepends macro risk warning badges to recommendations
