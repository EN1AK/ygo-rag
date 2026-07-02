## ADDED Requirements

### Requirement: LLM judge prompt is separate from final answer prompt

The system SHALL use a dedicated prompt for LLM rerank that is separate from the final answer synthesis prompt.

#### Scenario: LLM rerank without final LLM answer

- **GIVEN** LLM rerank is enabled
- **AND** final LLM answer generation is disabled
- **WHEN** a query is executed
- **THEN** the system uses the LLM judge prompt for ranking
- **AND** returns retrieval-only answer formatting for the final answer text

#### Scenario: LLM rerank with final LLM answer

- **GIVEN** LLM rerank is enabled
- **AND** final LLM answer generation is enabled
- **WHEN** a query is executed
- **THEN** the system first uses the LLM judge prompt for ranking
- **AND** then uses the final answer prompt with the reranked candidates

### Requirement: LLM judge output is structured

The system SHALL require LLM judge output to be parseable structured data containing card ids, scores, and reasons.

#### Scenario: Valid judge output

- **GIVEN** the LLM judge returns structured data for candidate cards
- **WHEN** the response is parsed
- **THEN** each accepted scored item includes card id, numeric score, and a concise reason

#### Scenario: Invalid judge output

- **GIVEN** the LLM judge response cannot be parsed as the required structure
- **WHEN** the response is processed
- **THEN** the system reports a clear LLM rerank parsing error

### Requirement: LLM judge reasons are exposed in candidate results

The system SHALL expose LLM judge reasons for candidates ranked by LLM rerank.

#### Scenario: Candidate ranked by LLM judge

- **GIVEN** LLM rerank assigns a reason to a candidate
- **WHEN** the candidate is returned to CLI or Web output
- **THEN** the candidate reason indicates LLM judge reranking
- **AND** includes or references the judge reason
