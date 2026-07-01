# retrieval-ranking Specification

## Purpose

Define the current retrieval and ranking behavior for similar card effect queries.

## Requirements

### Requirement: Sparse retrieval tokenizes Chinese and ASCII text

The system SHALL tokenize retrieval text into Chinese characters, adjacent Chinese bigrams, and ASCII alphanumeric terms.

#### Scenario: Sparse query contains Chinese effect terms

- **GIVEN** a query containing Chinese effect terms
- **WHEN** sparse retrieval runs
- **THEN** cards sharing those character and bigram terms can receive positive sparse scores

### Requirement: Sparse retrieval boosts exact card name matches

The system SHALL add a strong score boost when a card name appears as a substring of the query.

#### Scenario: Query mentions a card name

- **GIVEN** a card named `我身作盾`
- **AND** the query contains `我身作盾`
- **WHEN** sparse retrieval runs
- **THEN** that card receives an exact-name boost

### Requirement: Query may be expanded with referenced card effect

The system SHALL append the referenced card name and description to the retrieval query when the input query contains an exact card name with length at least two.

#### Scenario: Query contains one known card name

- **GIVEN** loaded cards include `我身作盾`
- **AND** the query contains `我身作盾`
- **WHEN** retrieval runs
- **THEN** the retrieval query includes the original query, the reference card name, and the reference card effect text

#### Scenario: Query contains multiple known card names

- **GIVEN** multiple loaded card names appear in the query
- **WHEN** retrieval query expansion runs
- **THEN** the longest matching card name is selected as the reference card

### Requirement: Hybrid ranking uses reciprocal rank fusion

The system SHALL merge sparse and optional dense ranked lists using reciprocal rank fusion.

#### Scenario: Candidate appears in multiple ranked lists

- **GIVEN** the same card appears in dense and sparse candidates
- **WHEN** fusion runs
- **THEN** the card is deduplicated by card id
- **AND** its fusion score accumulates contributions from each ranked list

### Requirement: Original candidate score is used as a tie breaker

The system SHALL use the original candidate score as a tie breaker when fusion scores are equal.

#### Scenario: Fusion scores tie

- **GIVEN** two candidates have equal reciprocal-rank scores
- **WHEN** results are sorted
- **THEN** the candidate with the higher non-negative original score ranks first
- **AND** card id is used as the final deterministic tie breaker

### Requirement: Reranking is optional

The system SHALL rerank fused candidates only when reranking is requested.

#### Scenario: Reranking requested

- **GIVEN** fused candidates exist
- **WHEN** reranking is enabled
- **THEN** candidates are passed to the configured reranker
- **AND** final results are sorted by reranker score descending and card id ascending

### Requirement: CUDA embedding uses subprocess reranker by default

The system SHALL use a subprocess reranker when reranking is requested and the embedding device starts with `cuda`.

#### Scenario: CUDA embedding with no explicit reranker device

- **GIVEN** embedding device is `cuda`
- **AND** no explicit reranker device is configured
- **WHEN** reranking is requested
- **THEN** reranking runs in a subprocess with reranker device `auto`

#### Scenario: CUDA embedding with explicit reranker device

- **GIVEN** embedding device is `cuda`
- **AND** an explicit reranker device is configured
- **WHEN** reranking is requested
- **THEN** the subprocess reranker uses the explicit reranker device

## Open Questions

- Should the reference card itself be excluded from similar-card results?
- Should sparse name boosting be configurable?
- Should retrieval distinguish primary effect intent from incidental shared terms such as cost or "无效并破坏"?
- Should reranker input use a structured effect profile instead of raw text?
