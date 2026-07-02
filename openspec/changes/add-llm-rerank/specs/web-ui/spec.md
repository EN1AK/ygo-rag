## ADDED Requirements

### Requirement: Web UI can select LLM rerank

The system SHALL expose an LLM rerank option in the local Web UI.

#### Scenario: User selects LLM rerank in browser

- **GIVEN** the Web UI is open
- **WHEN** the user selects LLM rerank and submits a query
- **THEN** the browser sends a query API request indicating LLM rerank selection

### Requirement: Web API accepts LLM rerank selection

The system SHALL parse LLM rerank selection from `POST /api/query`.

#### Scenario: API request selects LLM rerank

- **GIVEN** a JSON query request includes LLM rerank selection
- **WHEN** `POST /api/query` is called
- **THEN** the parsed query request selects LLM rerank for candidate ordering

### Requirement: Web UI separates LLM rerank from LLM answer generation

The system SHALL present LLM rerank as a separate option from final DeepSeek answer generation.

#### Scenario: LLM rerank enabled and LLM answer disabled

- **GIVEN** the user enables LLM rerank
- **AND** the user disables final LLM answer generation
- **WHEN** the query is submitted
- **THEN** the API request enables LLM rerank
- **AND** does not enable final LLM answer generation

### Requirement: Web output shows LLM judge reasons

The system SHALL display candidate reasons returned by LLM rerank.

#### Scenario: LLM judge reason exists

- **GIVEN** a query response contains a candidate reason from LLM judge reranking
- **WHEN** the Web UI renders results
- **THEN** the candidate displays that reason in the result card
