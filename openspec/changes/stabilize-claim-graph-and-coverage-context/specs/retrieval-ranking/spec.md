## ADDED Requirements

### Requirement: Adjudication retrieval queries distinguish FAQ and rules intent
The system SHALL guide and validate adjudication retrieval queries according to the target evidence source.

#### Scenario: FAQ query keeps card-specific terms
- **GIVEN** the planner calls a FAQ search tool
- **WHEN** it builds the query
- **THEN** the query MAY include card names, effect names, and concrete card interactions
- **AND** the search SHALL preserve resolved card id filters when provided

#### Scenario: Rules query abstracts card names
- **GIVEN** the planner calls `search_rules`
- **WHEN** it builds the query
- **THEN** the query SHALL focus on general OCG mechanics rather than card names as primary terms
- **AND** the planner prompt SHALL instruct the LLM to translate card-specific facts into mechanism-focused language

#### Scenario: Rules query is card-name-heavy
- **GIVEN** a `search_rules` query primarily consists of resolved card names and broad card-specific effect terms
- **WHEN** the tool validates the query
- **THEN** the system SHOULD reject or warn about the query
- **AND** the observation SHOULD ask for a mechanism-focused rewrite
