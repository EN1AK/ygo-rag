## ADDED Requirements

### Requirement: Runtime settings include persistent ruling index paths
The system SHALL expose runtime settings for persistent FAQ/supplement and OCG rules index storage.

#### Scenario: Default ruling index settings
- **GIVEN** no ruling index environment variables are set
- **WHEN** settings are loaded
- **THEN** the FAQ/supplement index path defaults under the data directory
- **AND** the OCG rules index path defaults under the data directory
- **AND** default collection names are configured for FAQ/supplement and rules indexes

#### Scenario: Custom ruling index settings
- **GIVEN** ruling index path or collection name environment variables are set
- **WHEN** settings are loaded
- **THEN** the configured values are used by index build and retrieval services

### Requirement: Runtime settings control ruling retrieval limits
The system SHALL expose runtime settings for ruling sparse recall, dense recall, fused top-k, and final evidence top-k.

#### Scenario: Default ruling retrieval limits
- **GIVEN** no ruling retrieval limit variables are set
- **WHEN** settings are loaded
- **THEN** FAQ/supplement and rules retrieval use documented default top-k values

#### Scenario: Custom ruling retrieval limits
- **GIVEN** ruling retrieval limit environment variables are set
- **WHEN** ruling retrieval runs
- **THEN** sparse recall, dense recall, fused candidate count, and final evidence count use the configured values

### Requirement: Runtime settings control ruling rerank provider
The system SHALL allow ruling evidence rerank provider selection among no rerank, local rerank, and LLM rerank.

#### Scenario: No ruling rerank selected
- **GIVEN** no ruling rerank provider is selected
- **WHEN** persistent ruling retrieval runs
- **THEN** fused candidate ordering is used without reranker scoring

#### Scenario: Local ruling rerank selected
- **GIVEN** local ruling rerank is selected
- **WHEN** persistent ruling retrieval runs
- **THEN** the configured local reranker is used on ruling evidence candidates

#### Scenario: LLM ruling rerank selected
- **GIVEN** LLM ruling rerank is selected
- **AND** DeepSeek credentials are configured
- **WHEN** persistent ruling retrieval runs
- **THEN** the DeepSeek-compatible LLM judge reranks ruling evidence candidates
