# indexing-and-embeddings Specification

## Purpose

Define the current behavior for converting cards into retrieval documents, embedding them, and persisting dense vectors in Chroma.

## Requirements

### Requirement: Card effect text is normalized for retrieval documents

The system SHALL collapse whitespace and line breaks in card descriptions when constructing retrieval text.

#### Scenario: Description contains line breaks

- **GIVEN** a card description containing CRLF, LF, and repeated whitespace
- **WHEN** the description is normalized
- **THEN** the normalized text is a single readable line with collapsed whitespace

### Requirement: Retrieval documents include card name, effect text, and metadata

The system SHALL convert each card into a retrieval document whose page content includes card name and effect text.

#### Scenario: Convert card to document

- **GIVEN** a card with name, description, id, and metadata
- **WHEN** the card is converted to a retrieval document
- **THEN** page content starts with `卡名：`
- **AND** page content includes `效果：`
- **AND** metadata includes card id, name, original description, type, race, attribute, attack, defense, level, and category

### Requirement: Chroma index can be built from card documents

The system SHALL build a persistent Chroma collection from retrieval documents using configured embeddings.

#### Scenario: Build index

- **GIVEN** loaded card records
- **WHEN** the `build-index` command runs
- **THEN** cards are converted to retrieval documents
- **AND** embeddings are loaded using the configured embedding model and embedding device
- **AND** documents are added to the configured Chroma collection in batches
- **AND** progress is printed after each batch

### Requirement: Build index can limit and reset input

The system SHALL support limiting the number of cards indexed and resetting the target Chroma collection.

#### Scenario: Build limited index

- **GIVEN** `--limit` is provided
- **WHEN** the index is built
- **THEN** only the first requested number of loaded cards are indexed

#### Scenario: Reset index

- **GIVEN** `--reset` is provided
- **WHEN** the index is built
- **THEN** the existing Chroma collection is deleted if possible before documents are added

### Requirement: Embedding backend depends on device

The system SHALL use `sentence-transformers` by default and a transformers-based backend when the embedding device starts with `cuda`.

#### Scenario: CUDA embedding backend

- **GIVEN** the embedding device starts with `cuda`
- **WHEN** embeddings are generated
- **THEN** the system loads tokenizer and model through `transformers`
- **AND** normalized CLS-token vectors are returned

#### Scenario: Non-CUDA embedding backend

- **GIVEN** the embedding device does not start with `cuda`
- **WHEN** embeddings are generated
- **THEN** the system loads `SentenceTransformer`
- **AND** returns normalized vectors from `model.encode`

### Requirement: Dense retrieval skips partial Chroma index

The system SHALL skip dense retrieval when the Chroma collection contains fewer documents than the loaded database.

#### Scenario: Chroma index is partial

- **GIVEN** semantic retrieval is requested
- **AND** Chroma document count is less than the number of loaded cards
- **WHEN** a query is executed
- **THEN** dense retrieval is not used
- **AND** the response includes a warning describing the count mismatch

## Open Questions

- Should the index store a manifest containing database hash, embedding model, and schema version?
- Should Chroma collection names be configurable?
- Should partial indexes be queryable when the user explicitly opts in?
