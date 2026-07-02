## ADDED Requirements

### Requirement: Rerank provider can be configured

The system SHALL allow rerank provider selection among no rerank, local rerank, and LLM rerank.

#### Scenario: No provider selected

- **GIVEN** no rerank provider is selected by CLI, Web, or environment
- **WHEN** a query is executed
- **THEN** no reranker is configured

#### Scenario: Local provider selected

- **GIVEN** local rerank provider is selected
- **WHEN** a query is executed
- **THEN** the system configures the existing local bge reranker behavior

#### Scenario: LLM provider selected

- **GIVEN** LLM rerank provider is selected
- **WHEN** a query is executed
- **THEN** the system configures LLM rerank using the DeepSeek-compatible LLM settings

### Requirement: LLM rerank requires DeepSeek credentials

The system SHALL require `DEEPSEEK_API_KEY` when LLM rerank is selected.

#### Scenario: LLM rerank without API key

- **GIVEN** LLM rerank is selected
- **AND** `DEEPSEEK_API_KEY` is not set
- **WHEN** a query is executed
- **THEN** the system raises an error stating that DeepSeek credentials are required for LLM rerank

### Requirement: LLM rerank candidate count is bounded

The system SHALL enforce a maximum number of candidates sent to LLM rerank.

#### Scenario: Candidate count exceeds configured maximum

- **GIVEN** LLM rerank is selected
- **AND** the fused candidate pool exceeds the configured LLM rerank maximum
- **WHEN** reranking runs
- **THEN** only the allowed number of candidates is sent to the LLM judge
- **AND** excluded candidates are not sent in the judge prompt

### Requirement: DeepSeek call timeout can be configured

The system SHALL allow DeepSeek request timeout configuration through runtime settings.

#### Scenario: Timeout is not configured

- **GIVEN** no DeepSeek timeout setting is configured
- **WHEN** DeepSeek LLM integration is created
- **THEN** the request timeout defaults to 60 seconds

#### Scenario: Timeout is configured

- **GIVEN** a DeepSeek timeout setting is configured
- **WHEN** DeepSeek LLM integration is created
- **THEN** the configured timeout is passed to the chat model client

## Open Questions

- What environment variable name should configure the default rerank provider?
- What default maximum should be used for LLM rerank candidates?
