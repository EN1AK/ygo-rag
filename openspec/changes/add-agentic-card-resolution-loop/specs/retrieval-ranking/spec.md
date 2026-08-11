## ADDED Requirements

### Requirement: Card retrieval supports agentic candidate lookup
The system SHALL expose card retrieval results in a structured form suitable for agentic card resolution.

#### Scenario: Agent requests card candidates
- **GIVEN** an extracted possible card-name string
- **WHEN** the card candidate provider is called
- **THEN** it returns local candidates with card id, card name, score, source, match reason, and enough card text for disambiguation

#### Scenario: Candidate lookup has no strong match
- **GIVEN** local retrieval returns only low-confidence candidates
- **WHEN** the agent evaluates the candidate list
- **THEN** the candidate provider preserves the ranked candidates
- **AND** the agent may mark the mention unresolved rather than selecting a weak candidate

### Requirement: LLM card selection is constrained to retrieved candidates
The system SHALL restrict LLM-assisted card selection to the local candidates returned by retrieval.

#### Scenario: LLM selection prompt is built
- **GIVEN** an ambiguous mention and local card candidates
- **WHEN** the LLM selection prompt is built
- **THEN** it includes only the retrieved candidate ids, names, scores, match reasons, and card text
- **AND** it instructs the LLM not to introduce candidates outside the provided set

#### Scenario: LLM returns an unknown selected id
- **GIVEN** the candidate set contains card ids `1` and `2`
- **WHEN** the LLM selection output contains card id `999`
- **THEN** the unknown id is rejected
- **AND** the mention remains ambiguous or unresolved
