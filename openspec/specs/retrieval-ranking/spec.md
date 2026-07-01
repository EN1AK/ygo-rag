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

### Requirement: LLM rerank is an optional rerank provider

The system SHALL support LLM rerank as an optional rerank provider without removing the existing local reranker.

#### Scenario: LLM rerank selected

- **GIVEN** fused candidates exist
- **AND** LLM rerank is selected
- **WHEN** retrieval runs
- **THEN** candidates are passed to the LLM rerank provider
- **AND** final results are sorted by LLM judge score descending and card id ascending

#### Scenario: Local rerank selected

- **GIVEN** fused candidates exist
- **AND** local rerank is selected
- **WHEN** retrieval runs
- **THEN** the existing local reranker behavior is used

#### Scenario: No rerank selected

- **GIVEN** fused candidates exist
- **AND** no rerank provider is selected
- **WHEN** retrieval runs
- **THEN** fused candidate ordering is returned without reranker scoring

### Requirement: LLM rerank uses only retrieved candidates

The system SHALL restrict LLM rerank input to the candidate cards produced by the retrieval pipeline.

#### Scenario: LLM rerank prompt is built

- **GIVEN** a query and fused candidates
- **WHEN** the LLM rerank prompt is built
- **THEN** the prompt includes only the candidate card ids, names, source texts, and retrieval scores from the provided candidates
- **AND** the prompt instructs the LLM not to introduce cards outside that candidate set

### Requirement: LLM rerank rejects unknown candidate ids

The system SHALL not accept LLM judge rankings for card ids that were not present in the input candidate set.

#### Scenario: LLM returns an unknown card id

- **GIVEN** LLM rerank input contains candidate ids `1` and `2`
- **WHEN** the LLM judge output includes card id `999`
- **THEN** the unknown id is rejected
- **AND** it is not included in final ranked results

### Requirement: LLM rerank preserves unscored candidates deterministically

The system SHALL preserve candidates not scored by the LLM judge after scored candidates in deterministic order.

#### Scenario: LLM omits a candidate

- **GIVEN** three candidates are sent to LLM rerank
- **AND** the LLM judge returns scores for only two of them
- **WHEN** final results are built
- **THEN** scored candidates appear first ordered by judge score
- **AND** the unscored candidate appears after scored candidates using its prior candidate order as tie breaker

### Requirement: LLM rerank falls back on judge failure

The system SHALL fall back to the original candidate order when the LLM judge call fails or returns unparsable output.

#### Scenario: LLM judge times out

- **GIVEN** LLM rerank is selected
- **WHEN** the LLM judge call times out
- **THEN** the original candidate order is returned
- **AND** the response includes a warning that LLM rerank failed and hybrid candidate order was used

#### Scenario: LLM judge returns malformed output

- **GIVEN** LLM rerank is selected
- **WHEN** the LLM judge output cannot be parsed
- **THEN** the original candidate order is returned
- **AND** the response includes a warning that LLM rerank failed and hybrid candidate order was used

## Open Questions

- Should the reference card itself be excluded from similar-card results?
- Should sparse name boosting be configurable?
- Should retrieval distinguish primary effect intent from incidental shared terms such as cost or "无效并破坏"?
- Should reranker input use a structured effect profile instead of raw text?
