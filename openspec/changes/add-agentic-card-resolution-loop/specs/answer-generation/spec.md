## ADDED Requirements

### Requirement: Adjudication answers consume final resolved-card state
The system SHALL generate adjudication evidence and answers from the final card-resolution loop state.

#### Scenario: Card resolution retries change selected cards
- **GIVEN** the card-resolution loop resolves a card after targeted retry
- **WHEN** ruling evidence retrieval runs
- **THEN** the newly resolved card id is included in direct and related ruling retrieval

#### Scenario: Card mentions remain unresolved
- **GIVEN** the card-resolution loop leaves one or more mentions unresolved
- **WHEN** the final adjudication answer is generated
- **THEN** the answer prompt includes the unresolved mentions through coverage or trace context
- **AND** the answer lists the resulting uncertainty if it affects the ruling

### Requirement: Adjudication answers include card-resolution trace context
The system SHALL provide concise card-resolution trace context to the final answer prompt when agent tracing is available.

#### Scenario: Final answer prompt is built
- **GIVEN** the agent has card-resolution trace entries
- **WHEN** the final adjudication answer prompt is built
- **THEN** the prompt includes a concise summary of resolved, ambiguous, and unresolved card mentions
- **AND** the prompt does not expose unrelated low-confidence candidate noise unless full trace is requested
