## ADDED Requirements

### Requirement: Persistent ruling retrieval is staged
The system SHALL retrieve persistent ruling evidence in separate direct, related, analogous, supplement, and rules stages.

#### Scenario: Resolved card ids exist
- **GIVEN** card resolution produced one or more resolved card ids
- **WHEN** persistent evidence retrieval runs
- **THEN** the agent SHALL run a direct card evidence stage for those card ids
- **AND** the agent SHALL run a related-card evidence stage for refer-based rulings
- **AND** the agent SHALL run separate analogous FAQ and rules stages for mechanism retrieval

#### Scenario: No resolved card ids exist
- **GIVEN** card resolution produced no resolved card ids
- **WHEN** persistent evidence retrieval runs
- **THEN** direct and related stages SHALL be skipped
- **AND** analogous FAQ and rules stages MAY still run using the scenario query

### Requirement: Evidence relationship labels are preserved
The system SHALL preserve evidence relationship labels from persistent retrieval metadata.

#### Scenario: FAQ primary card matches resolved card
- **GIVEN** a persistent FAQ document has a `primary_card_ids` value matching a resolved card id
- **WHEN** the document is returned as evidence
- **THEN** its relationship SHALL include `direct_card`

#### Scenario: FAQ refer card matches resolved card
- **GIVEN** a persistent FAQ document has a `refer_card_ids` value matching a resolved card id
- **WHEN** the document is returned as evidence
- **THEN** its relationship SHALL include `related_card`

#### Scenario: Rule passage is returned
- **GIVEN** a persistent rule passage is returned as evidence
- **WHEN** the evidence object is built
- **THEN** its relationship SHALL be `rule_document`

#### Scenario: Card supplement is returned
- **GIVEN** a persistent card supplement is returned as evidence
- **WHEN** the evidence object is built
- **THEN** its relationship SHALL be `card_supplement`

### Requirement: Direct evidence is not flattened into analogous evidence
The system SHALL NOT label direct or related persistent FAQ evidence as only analogous when card-id metadata proves a stronger relationship.

#### Scenario: Direct FAQ appears in analogous search results
- **GIVEN** an analogous search returns a FAQ whose primary card id matches a resolved card id
- **WHEN** evidence is deduplicated
- **THEN** the final relationship SHALL retain `direct_card`
- **AND** `analogous` MAY be included only as an additional relationship if the evidence also came from the analogous stage

### Requirement: Evidence stage trace is exposed
The system SHALL expose retrieval stage metadata in the agent trace.

#### Scenario: Persistent retrieval completes
- **WHEN** the agent finishes persistent evidence retrieval
- **THEN** the trace SHALL list each retrieval stage, requested card ids, query text or query summary, returned evidence ids, and relationship labels

### Requirement: Final answer prioritizes staged evidence
The system SHALL present evidence to final answer generation with direct and related evidence distinguishable from analogous and rules evidence.

#### Scenario: Final prompt is built
- **GIVEN** evidence includes direct, related, analogous, supplement, and rules items
- **WHEN** the final adjudication answer prompt is built
- **THEN** each evidence item SHALL include its relationship label
- **AND** answer-generation rules SHALL continue to prioritize direct FAQ, then related FAQ, then supplement and card text, then analogous FAQ, then rules
