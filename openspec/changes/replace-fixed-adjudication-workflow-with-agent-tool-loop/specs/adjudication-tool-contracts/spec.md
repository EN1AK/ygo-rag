## ADDED Requirements

### Requirement: Adjudication tools have strict schemas
The system SHALL define each adjudication tool with a stable name, JSON argument schema, structured observation schema, and validation errors.

#### Scenario: Invalid tool arguments
- **GIVEN** the planner emits an allowed tool name with invalid arguments
- **WHEN** the tool dispatcher validates the action
- **THEN** the tool SHALL NOT run
- **AND** the observation SHALL contain a validation error visible to the next planner iteration

#### Scenario: Unknown tool name
- **GIVEN** the planner emits a tool name that is not in the allowed tool set
- **WHEN** the dispatcher receives the action
- **THEN** no retrieval or mutation tool SHALL run
- **AND** the loop SHALL record an invalid-action observation

### Requirement: Card tools preserve local validation gates
The system SHALL use local card database lookup, card RAG, verified aliases, and absolute validity gates when resolving card mentions or searching card candidates.

#### Scenario: Resolve cards tool validates candidates
- **GIVEN** the planner calls `resolve_cards` with mention strings
- **WHEN** card candidates are retrieved
- **THEN** each candidate SHALL include validation status and validation reason
- **AND** unresolved or ambiguous mentions SHALL remain unresolved or ambiguous unless a candidate passes the validity gate

#### Scenario: Search card tool returns candidates only
- **GIVEN** the planner calls `search_card`
- **WHEN** card candidates are returned
- **THEN** the observation SHALL identify them as candidates
- **AND** the tool MUST NOT mark them as resolved unless the resolve operation validates them

### Requirement: Ruling tools preserve evidence relationships
The system SHALL expose separate tools for direct FAQ, related FAQ, analogous FAQ, card supplement, and rules retrieval while preserving evidence relationship labels.

#### Scenario: Direct FAQ tool runs
- **GIVEN** the planner calls `search_direct_faq` with resolved card ids
- **WHEN** FAQ evidence is found through primary card metadata
- **THEN** returned evidence SHALL include the `direct_card` relationship

#### Scenario: Related FAQ tool runs
- **GIVEN** the planner calls `search_related_faq` with resolved card ids
- **WHEN** FAQ evidence is found through refer metadata
- **THEN** returned evidence SHALL include the `related_card` relationship

#### Scenario: Analogous FAQ tool runs
- **GIVEN** the planner calls `search_analogous_faq` with a query
- **WHEN** FAQ evidence is retrieved by query similarity or sparse matching
- **THEN** returned evidence SHALL include the `analogous` relationship unless stronger card metadata proves `direct_card` or `related_card`

#### Scenario: Rules tool runs
- **GIVEN** the planner calls `search_rules` with a query
- **WHEN** rules passages are returned
- **THEN** returned evidence SHALL include the `rule_document` relationship

### Requirement: Tools add durable evidence to the evidence pool
The system SHALL store evidence returned by retrieval tools in a shared evidence pool with ids, relationships, source, stage, query, and metadata.

#### Scenario: Duplicate evidence is returned
- **GIVEN** the same evidence id is returned by multiple tools
- **WHEN** the evidence pool is updated
- **THEN** the evidence SHALL be deduplicated
- **AND** stronger relationships such as `direct_card` and `related_card` SHALL be retained alongside retrieval stages

### Requirement: Coverage critique identifies next evidence gaps
The system SHALL provide a critique operation that summarizes unresolved cards, ambiguous cards, unanswered questions, weak evidence relationships, and possible next searches.

#### Scenario: Evidence is incomplete
- **GIVEN** the evidence pool does not cover all adjudication questions
- **WHEN** the planner calls `critique_coverage`
- **THEN** the observation SHALL list evidence gaps and unresolved card state
- **AND** it SHALL NOT fabricate evidence ids or card ids
