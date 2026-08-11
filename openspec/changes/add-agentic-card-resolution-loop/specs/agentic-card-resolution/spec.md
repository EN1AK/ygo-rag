## ADDED Requirements

### Requirement: Agent extracts possible card-name mentions
The system SHALL extract possible card-name strings from adjudication scenarios before resolving cards.

#### Scenario: LLM extraction is available
- **GIVEN** adjudication runs with an LLM configured
- **WHEN** the agent begins card resolution
- **THEN** the agent asks the LLM for structured possible card-name strings and context spans
- **AND** the agent does not accept card ids from the LLM extraction output

#### Scenario: LLM extraction fails
- **GIVEN** LLM card-name extraction is enabled
- **WHEN** the LLM output is missing, malformed, or unparsable
- **THEN** the agent falls back to deterministic mention extraction
- **AND** the agent records a warning in the card-resolution trace

#### Scenario: LLM is not configured
- **GIVEN** adjudication runs without an LLM
- **WHEN** the agent begins card resolution
- **THEN** the agent uses deterministic mention extraction

### Requirement: Agent retrieves local card candidates for extracted mentions
The system SHALL use the local card database and card retrieval pipeline to retrieve candidates for each extracted mention.

#### Scenario: Mention is searched
- **GIVEN** an extracted possible card-name string
- **WHEN** card resolution runs
- **THEN** the agent retrieves top-k local card candidates for that string
- **AND** each candidate includes card id, card name, score, source, and match reason

#### Scenario: LLM proposes unknown card text
- **GIVEN** the LLM extracts a card-name string that does not match any local card candidate above threshold
- **WHEN** candidate retrieval completes
- **THEN** the mention is marked unresolved
- **AND** no card id is invented for that mention

### Requirement: Agent selects cards only from local candidates
The system SHALL resolve a card mention only to a card id present in the local retrieved candidates.

#### Scenario: High-confidence rule match exists
- **GIVEN** a mention has a local candidate with exact or normalized substring match above the confidence threshold
- **WHEN** candidate selection runs
- **THEN** the agent resolves the mention to that candidate without needing LLM selection

#### Scenario: Candidate set is ambiguous
- **GIVEN** multiple local candidates are within the ambiguity margin
- **WHEN** LLM selection is configured
- **THEN** the agent may ask the LLM to choose among only those local candidates
- **AND** the agent rejects any selected id not present in the candidate set

#### Scenario: LLM selection fails
- **GIVEN** LLM selection is configured for an ambiguous mention
- **WHEN** the LLM returns malformed output or an unknown id
- **THEN** the mention remains ambiguous
- **AND** the agent records a warning

### Requirement: Agent retries unresolved and ambiguous card mentions
The system SHALL run bounded targeted card-search retries for unresolved or ambiguous mentions before retrieving ruling evidence.

#### Scenario: Unresolved mentions remain after first pass
- **GIVEN** at least one extracted mention is unresolved
- **WHEN** retry budget remains
- **THEN** the agent forms targeted search queries from the original mention, neighboring scenario text, and action/effect terms
- **AND** the agent runs local card candidate retrieval again

#### Scenario: Ambiguous mentions remain after first pass
- **GIVEN** at least one extracted mention is ambiguous
- **WHEN** retry budget remains
- **THEN** the agent expands the mention query using scenario context
- **AND** the agent attempts selection again against local candidates

#### Scenario: Retry budget is exhausted
- **GIVEN** unresolved or ambiguous mentions remain
- **WHEN** the retry budget is exhausted
- **THEN** the agent proceeds with resolved cards only
- **AND** the coverage report lists unresolved or ambiguous card mentions

### Requirement: Agent traces card-resolution loop decisions
The system SHALL expose trace entries for the card-resolution loop.

#### Scenario: Card resolution completes
- **WHEN** the agent finishes card resolution
- **THEN** the trace includes extracted mentions, extraction source, candidate ids, selected ids, unresolved mentions, ambiguous mentions, retry count, and warnings

#### Scenario: Persistent ruling retrieval runs after card resolution
- **GIVEN** card resolution produced resolved card ids
- **WHEN** ruling evidence retrieval runs
- **THEN** the retrieval query uses the resolved card ids from the final card-resolution loop state
