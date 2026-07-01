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

## Open Questions

- Should `.env` files be supported explicitly, or should environment variables remain the only credential mechanism?
- Should model paths and model names be validated at startup or only when used?
- Should runtime settings be exposed through the Web UI for inspection?
