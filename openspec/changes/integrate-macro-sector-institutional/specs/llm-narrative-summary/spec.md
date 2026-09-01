## Purpose

Provides natural language market commentary and strategic insight synthesis using OpenAI-compatible endpoints with configurable 3-tier fallback and graceful degradation.

## ADDED Requirements

### Requirement: 3-Tier LLM Fallback Execution
The system SHALL support configuring up to three distinct LLM providers (Primary, Fallback 1, Fallback 2) with independent `base_url`, `model`, and `api_key` settings, sequentially attempting calls in order of priority upon network failure, timeout, or API error.

#### Scenario: Primary LLM Success
- **WHEN** Primary LLM returns a valid HTTP 200 completion response within the configured timeout
- **THEN** system adopts the generated commentary text and avoids invoking any fallback providers

#### Scenario: Primary Failure and Fallback 1 Activation
- **WHEN** Primary LLM encounters timeout, HTTP 429 (Rate Limit), or HTTP 5xx error
- **THEN** system logs a warning and automatically switches to Fallback 1 provider

#### Scenario: Cascading Failure to Fallback 2
- **WHEN** both Primary and Fallback 1 fail
- **THEN** system logs a warning and attempts generation via Fallback 2 provider

#### Scenario: Universal Fallback to Template
- **WHEN** all configured LLM providers fail or when LLM summarization is disabled
- **THEN** system logs an info message and generates a deterministic rule-based text summary without failing the pipeline

### Requirement: Ground-Truth Constrained Narrative Generation
The system SHALL format pre-calculated quantitative data (Macro regime, Sector leaders, Triple Resonance picks, Technical/LSTM scores) into a strictly grounded context prompt and mandate that the LLM summarize only factual inputs without inventing unverified numbers.

#### Scenario: Fact-based Commentary Output
- **WHEN** structured evaluation metrics are supplied to the LLM engine
- **THEN** LLM returns a concise 100-150 word summary highlighting macro risk, leading sectors, and priority stock rationale
