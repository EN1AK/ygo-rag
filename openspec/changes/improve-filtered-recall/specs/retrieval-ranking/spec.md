## ADDED Requirements

### Requirement: Structured-filtered retrieval uses a wider recall pool

The system SHALL use at least 100 candidates from each available retrieval source before hybrid fusion when structured query filters are applied.

#### Scenario: Sparse candidate ranks below final top-k

- **GIVEN** structured filters match a candidate card
- **AND** sparse retrieval ranks that candidate at position 37 for the filtered effect query
- **AND** final `top_k` is 10
- **WHEN** retrieval runs with structured filters
- **THEN** the candidate remains eligible for hybrid fusion or reranking because the sparse recall pool is at least 100

#### Scenario: Filtered set has fewer than 100 cards

- **GIVEN** structured filters match fewer than 100 cards
- **WHEN** retrieval runs
- **THEN** the system uses all matching filtered cards as the maximum recall pool for sparse retrieval

### Requirement: Dense retrieval accepts structured metadata filters

The system SHALL pass structured query filters to the dense retriever as vector-store metadata filters when semantic retrieval is enabled.

#### Scenario: Rank four XYZ dense query

- **GIVEN** parsed filters identify rank 4 XYZ monsters
- **AND** semantic retrieval is enabled
- **WHEN** dense retrieval runs
- **THEN** the dense retriever receives a metadata filter requiring monster cards, XYZ subtype, and rank 4

#### Scenario: Dense metadata filter fails on stale index

- **GIVEN** parsed filters are present
- **AND** semantic retrieval is enabled
- **AND** dense metadata-filtered search fails because the Chroma index lacks required metadata
- **WHEN** retrieval runs
- **THEN** the system falls back to the existing safe dense post-filter behavior
- **AND** the response includes a warning explaining that dense metadata filtering was not applied

### Requirement: Unfiltered retrieval keeps existing recall size

The system SHALL preserve existing sparse and dense retrieval candidate counts when structured query parsing produces no filters.

#### Scenario: Query has no structured filters

- **GIVEN** a query contains no recognized structured filters
- **WHEN** retrieval runs with `top_k` 10
- **THEN** sparse and dense retrieval use the existing unfiltered candidate count behavior
