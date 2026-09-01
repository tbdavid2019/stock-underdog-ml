# pipeline-orchestrator Specification

## Purpose
Orchestrates the end-to-end execution pipeline by separating concurrent network I/O retrieval from compute workloads and routing results to persistence and notification sinks.

## Requirements

### Requirement: Two-Stage Execution Pipeline
The system SHALL organize index analysis into two distinct stages: Stage 1 for concurrent I/O data prefetching and Stage 2 for compute-bound strategy execution and evaluation.

#### Scenario: Stage 1 prefetching completion
- **WHEN** an index component list is received
- **THEN** orchestrator concurrently downloads and caches all required market prices and fundamentals before strategy computation begins

#### Scenario: Stage 2 compute execution
- **WHEN** all data is loaded in memory/cache
- **THEN** orchestrator passes preloaded datasets to strategies for vectorized or parallel computation

### Requirement: Sinks Integration and Resiliency
The system SHALL route evaluation results to Supabase database and notification channels with comprehensive error isolation.

#### Scenario: Database sink failure isolation
- **WHEN** Supabase persistence encounters a network or schema error
- **THEN** system logs the error and continues to output console reports and send notifications without crashing the entire run
