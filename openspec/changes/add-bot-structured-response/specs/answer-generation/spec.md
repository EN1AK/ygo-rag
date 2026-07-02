## ADDED Requirements

### Requirement: Card message text is generated from retrieved candidate fields

The system SHALL generate per-card message text from structured retrieved candidate fields.

#### Scenario: Build card message

- **GIVEN** a retrieved candidate with card id, name, score, source text, and reason
- **WHEN** a structured card block is created
- **THEN** the message text includes rank, card name, card id, score, source text, and reason

### Requirement: Structured fields remain available separately from display text

The system SHALL include structured fields in each card block in addition to ready-to-send text.

#### Scenario: Bot consumes structured fields

- **GIVEN** a structured card block exists
- **WHEN** a bot reads the block
- **THEN** it can access card id, name, score, source text, and reason without parsing display text
