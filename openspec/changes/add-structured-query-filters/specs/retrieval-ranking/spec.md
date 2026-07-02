## ADDED Requirements

### Requirement: Retrieval applies structured candidate filters before ranking

The system SHALL apply parsed structured query filters to candidate cards before sparse, dense, hybrid, local rerank, or LLM rerank ranking is performed.

#### Scenario: Filtered sparse retrieval

- **GIVEN** parsed filters identify rank 4 XYZ monsters
- **WHEN** sparse retrieval runs
- **THEN** sparse scoring is limited to cards matching those filters

#### Scenario: Filtered dense retrieval

- **GIVEN** parsed filters identify rank 4 XYZ monsters
- **AND** semantic retrieval is enabled
- **WHEN** dense retrieval runs
- **THEN** dense candidates are limited to cards matching those filters

#### Scenario: Rerank receives filtered candidates

- **GIVEN** parsed filters identify rank 4 XYZ monsters
- **AND** reranking is enabled
- **WHEN** reranking runs
- **THEN** the reranker receives only candidates that match the parsed filters

### Requirement: Retrieval preserves existing behavior without filters

The system SHALL preserve existing sparse, dense, hybrid, and rerank behavior when structured query parsing produces no filters.

#### Scenario: Query has no recognized filters

- **GIVEN** a query contains no recognized structure constraints
- **WHEN** retrieval runs
- **THEN** sparse, dense, hybrid, and rerank behavior follows the existing unfiltered retrieval pipeline

### Requirement: Retrieval reports structured filter warnings

The system SHALL report a warning when parsed structured filters cannot be applied cleanly.

#### Scenario: Parsed filters match no cards

- **GIVEN** parsed filters are present
- **AND** no loaded card matches those filters
- **WHEN** retrieval runs
- **THEN** the response includes a warning describing the zero-match structured filter condition

#### Scenario: Dense index cannot enforce filters directly

- **GIVEN** parsed filters are present
- **AND** semantic retrieval is enabled
- **AND** the existing Chroma index cannot enforce the decoded metadata filter directly
- **WHEN** retrieval runs
- **THEN** dense results are constrained by an equivalent candidate allow-list or post-query filtering
- **AND** the final returned results still obey the parsed filters
