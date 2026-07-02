## ADDED Requirements

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
