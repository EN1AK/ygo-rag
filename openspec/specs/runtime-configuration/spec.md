# runtime-configuration Specification

## Purpose

Define the current runtime configuration behavior for paths, model names, device selection, Chroma storage, and DeepSeek credentials.

## Requirements

### Requirement: Runtime settings are read from environment variables

The system SHALL construct runtime settings from environment variables with documented defaults.

#### Scenario: Default settings

- **GIVEN** no relevant environment variables are set
- **WHEN** settings are loaded
- **THEN** data directory defaults to `data`
- **AND** cards database path defaults to `data/cards.cdb`
- **AND** Chroma persistence directory defaults to `data/chroma`
- **AND** embedding model defaults to `BAAI/bge-m3`
- **AND** reranker model defaults to `BAAI/bge-reranker-v2-m3`
- **AND** device defaults to `auto`
- **AND** DeepSeek base URL defaults to `https://api.deepseek.com`
- **AND** DeepSeek model defaults to `deepseek-chat`

### Requirement: Device settings can be shared or separated

The system SHALL use `RAG_DEVICE` as the default device for both embedding and reranking unless more specific variables are present.

#### Scenario: Shared CUDA device

- **GIVEN** `RAG_DEVICE` is set to `cuda`
- **WHEN** settings are loaded
- **THEN** embedding device is `cuda`
- **AND** reranker device is `cuda`
- **AND** reranker device is not marked as explicitly configured

#### Scenario: Explicit reranker device

- **GIVEN** `RAG_DEVICE` is `cuda`
- **AND** `RAG_RERANKER_DEVICE` is `auto`
- **WHEN** settings are loaded
- **THEN** embedding device remains `cuda`
- **AND** reranker device is `auto`
- **AND** reranker device is marked as explicitly configured

### Requirement: DeepSeek API key is environment-only

The system SHALL read the DeepSeek API key from `DEEPSEEK_API_KEY`.

#### Scenario: DeepSeek key is absent

- **GIVEN** `DEEPSEEK_API_KEY` is not set
- **WHEN** LLM answer generation is requested
- **THEN** the system raises an error stating that `DEEPSEEK_API_KEY` is required

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

- Should `.env` files be supported explicitly, or should environment variables remain the only credential mechanism?
- Should model paths and model names be validated at startup or only when used?
- Should runtime settings be exposed through the Web UI for inspection?
