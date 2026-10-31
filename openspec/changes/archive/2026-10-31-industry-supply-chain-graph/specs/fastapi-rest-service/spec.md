# Spec Delta

## ADDED Requirements

### Requirement: Industry Chain REST Endpoints
The system SHALL provide REST endpoints under `/api/v1/industry-chain` to query available industry themes and structured supply chain graph topologies with embedded quantitative indicators.

#### Scenario: Retrieve available industry themes
- **WHEN** client requests `GET /api/v1/industry-chain/themes`
- **THEN** system returns list of industry themes, stock counts, and sector classifications

#### Scenario: Query graph topology with quantitative metrics
- **WHEN** client requests `GET /api/v1/industry-chain/graph?theme=CPO%2F%E5%85%89%E4%BA%92%E8%81%AF`
- **THEN** system returns upstream, midstream, and downstream nodes populated with current price, return percentage, AI target price, and edges representing supply relationships
