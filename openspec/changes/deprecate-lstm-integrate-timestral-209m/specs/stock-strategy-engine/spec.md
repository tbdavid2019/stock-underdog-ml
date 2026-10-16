# Spec Delta: stock-strategy-engine

## REMOVED Requirements

### Requirement: LSTM Model Strategy Implementation
**Reason**: Outdated 1997 single-layer LSTM recurrent architecture experiences vanishing gradients, fails to capture multi-scale temporal context, and collapses to random-walk one-step persistence ($\hat{y}_{t+1} \approx y_t$).
**Migration**: Use the Mistral Timestral-209M foundation model strategy (`TimestralStrategy` in `models/timestral_model.py`) which provides multi-step, multi-quantile (P10/P50/P90) probabilistic forecasts.

## ADDED Requirements

### Requirement: Mistral Timestral Foundation Model Strategy Implementation
The system SHALL infer forward price trajectories and multi-quantile distributions (P10, P50, P90) for a stock using the Mistral Timestral-209M foundation model architecture and output predicted potential percentage and risk-reward profile.

#### Scenario: Timestral multi-quantile forecast generation
- **WHEN** historical daily price series (at least 60 trading days) is provided to `TimestralStrategy`
- **THEN** model outputs forward forecast horizon containing median target price (P50), upside ceiling (P90), downside floor (P10), and expected upside potential percentage

#### Scenario: Asymmetric risk-reward calculation
- **WHEN** Timestral produces valid P10, P50, and P90 price predictions
- **THEN** strategy computes risk-reward ratio `(P90 - current_price) / (current_price - P10)` and flags high-conviction asymmetric opportunities
