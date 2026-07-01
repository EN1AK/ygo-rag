# cli-interface Specification

## Purpose

Define the current command-line interface behavior for database management, indexing, querying, and starting the Web UI.

## Requirements

### Requirement: CLI exposes top-level commands

The system SHALL expose `download-db`, `inspect-db`, `build-index`, `query`, and `web` commands through `python -m rag_agent`.

#### Scenario: User asks for help

- **WHEN** `python -m rag_agent --help` is run
- **THEN** the help output lists `download-db`, `inspect-db`, `build-index`, `query`, and `web`

### Requirement: download-db downloads a cards database

The system SHALL download a database URL to a local destination path.

#### Scenario: Download database

- **GIVEN** a URL and output path
- **WHEN** `download-db` runs
- **THEN** the destination parent directory is created if necessary
- **AND** the URL content is copied to the destination file
- **AND** the command prints the destination path

### Requirement: inspect-db reports database counts

The system SHALL print the inspected database path and row counts for required tables.

#### Scenario: Inspect valid database

- **GIVEN** a valid database path
- **WHEN** `inspect-db --db <path>` runs
- **THEN** the output includes the database path
- **AND** row counts for `datas` and `texts`

### Requirement: build-index creates a local Chroma index

The system SHALL build a Chroma index from the selected database.

#### Scenario: Build index with defaults

- **GIVEN** a valid database path
- **WHEN** `build-index --db <path>` runs
- **THEN** cards are loaded
- **AND** embeddings are loaded
- **AND** Chroma documents are added in batches
- **AND** indexed progress and final indexed count are printed

### Requirement: query supports retrieval mode flags

The system SHALL allow query execution with optional semantic retrieval, reranking, LLM generation, top-k, and rerank-candidate settings.

#### Scenario: Query help

- **WHEN** `python -m rag_agent query --help` is run
- **THEN** help output includes `--semantic`
- **AND** `--rerank`
- **AND** `--llm`

#### Scenario: Sparse baseline query

- **GIVEN** a valid database path
- **WHEN** `query <text> --db <path>` runs without optional retrieval flags
- **THEN** the system returns a retrieval-only answer based on sparse retrieval

### Requirement: CLI errors return non-zero exit code

The system SHALL catch command exceptions, print an `error:` message to stderr, and return exit code `1`.

#### Scenario: Command raises exception

- **GIVEN** a command encounters an exception
- **WHEN** the CLI handles the exception
- **THEN** stderr begins with `error:`
- **AND** the command returns exit code `1`

### Requirement: web command starts the local Web UI

The system SHALL start the ASGI Web UI through `uvicorn`.

#### Scenario: Start Web UI

- **WHEN** `python -m rag_agent web --host 127.0.0.1 --port 7860` is run
- **THEN** the Web app is served on the requested host and port

## Open Questions

- Should CLI exit codes distinguish validation errors, runtime errors, and dependency errors?
- Should `query` support JSON output?
- Should `build-index` validate that the Chroma collection corresponds to the selected database after build?
