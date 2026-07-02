## ADDED Requirements

### Requirement: Query API can return bot-structured output

The system SHALL provide bot-oriented structured output in query API responses without removing existing response fields.

#### Scenario: API returns structured output

- **GIVEN** a valid query request
- **WHEN** `POST /api/query` returns successfully
- **THEN** the JSON response includes existing `answer`, `results`, and `warnings` fields
- **AND** includes a `structured` object for bot-oriented consumers

### Requirement: Structured output contains card-level blocks

The system SHALL split query results into card-level message blocks.

#### Scenario: Results contain candidate cards

- **GIVEN** query results contain candidate cards
- **WHEN** structured output is built
- **THEN** each candidate card has one `card` block
- **AND** each block includes card id, name, rank index, score, ready-to-send text, source fields, and truncation status

### Requirement: Structured output supports message length limiting

The system SHALL support an optional maximum text length for structured message blocks.

#### Scenario: Card block exceeds maximum length

- **GIVEN** a caller requests a maximum structured block length
- **AND** a generated card block text exceeds that maximum
- **WHEN** structured output is built
- **THEN** the block text is shortened to fit within the requested maximum
- **AND** the block marks `truncated` as true

### Requirement: Structured output is available without final LLM answer generation

The system SHALL produce structured card blocks even when final LLM answer generation is disabled.

#### Scenario: Retrieval-only query response

- **GIVEN** final LLM answer generation is disabled
- **WHEN** `POST /api/query` returns results
- **THEN** structured card blocks are built from retrieved candidate fields
