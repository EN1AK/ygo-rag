## ADDED Requirements

### Requirement: CLI builds persistent ruling indexes
The CLI SHALL expose commands for building persistent FAQ/supplement and OCG rules indexes from local source files.

#### Scenario: Help lists ruling index build command
- **WHEN** `python -m rag_agent --help` is run
- **THEN** the help output lists a command for building ruling indexes

#### Scenario: Build ruling indexes
- **GIVEN** a local cards database, ygocdb cache directory, and OCG rules EPUB are available
- **WHEN** the user runs the ruling index build command
- **THEN** FAQ/supplement and rules documents are built from local sources
- **AND** embeddings are written to persistent vector collections
- **AND** progress and final document counts are printed

#### Scenario: Build FAQ index only
- **GIVEN** the user selects only FAQ/supplement indexing
- **WHEN** the ruling index build command runs
- **THEN** only the FAQ/supplement collection and manifest are updated

#### Scenario: Build rules index only
- **GIVEN** the user selects only OCG rules indexing
- **WHEN** the ruling index build command runs
- **THEN** only the rules collection and manifest are updated

### Requirement: CLI inspects persistent ruling indexes
The CLI SHALL expose a command for inspecting persistent ruling index status.

#### Scenario: Inspect existing indexes
- **GIVEN** persistent ruling indexes and manifests exist
- **WHEN** the user runs the ruling index inspect command
- **THEN** the output includes collection names, manifest status, document counts, source counts, and embedding model

#### Scenario: Inspect missing indexes
- **GIVEN** a persistent ruling index is missing
- **WHEN** the user runs the ruling index inspect command
- **THEN** the output reports the missing index without raising an unhandled exception

### Requirement: CLI queries persistent ruling indexes
The CLI SHALL expose a command for testing FAQ/rules retrieval directly.

#### Scenario: Query ruling evidence
- **GIVEN** persistent ruling indexes exist
- **WHEN** the user runs a ruling query command with text
- **THEN** the output lists retrieved evidence ids, kinds, relationships, scores, titles, and snippets

#### Scenario: Query with LLM rerank
- **GIVEN** persistent ruling indexes exist
- **AND** DeepSeek credentials are configured
- **WHEN** the user runs a ruling query command with LLM rerank enabled
- **THEN** retrieved ruling evidence candidates are reranked by the LLM judge before output
