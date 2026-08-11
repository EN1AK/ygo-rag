## ADDED Requirements

### Requirement: Adjudication answers cite stable coverage ids
The system SHALL require agent adjudication final answers to cite stable issue or claim ids from the latest accepted coverage graph.

#### Scenario: Answer cites supported negative issue
- **GIVEN** the coverage graph contains a supported issue with negative polarity
- **WHEN** `submit_ruling` validates an answer that cites that issue id
- **THEN** the answer SHALL be accepted as grounded for a negative conclusion
- **AND** the answer SHALL cite evidence ids that support the negative issue

#### Scenario: Answer cites stale claim id
- **GIVEN** a previous critique used a claim id that no longer exists in the latest graph
- **WHEN** `submit_ruling` validates an answer citing that stale id
- **THEN** the submission SHALL be rejected
- **AND** the validation error SHALL identify the stale or unknown id

### Requirement: Budget-limited answers remain structured
The system SHALL return structured adjudication JSON when the agent reaches a budget limit and has enough evidence to state uncertainty.

#### Scenario: Budget exhausted with unresolved claims
- **GIVEN** the agent has unresolved or unsupported coverage graph nodes
- **AND** a planner budget is exhausted before accepted submission
- **WHEN** the final answer is produced
- **THEN** the answer SHALL include uncertainties that reference unresolved issue or claim ids
- **AND** warnings SHALL state which budget was exhausted

#### Scenario: Hard contradiction supports a negative conclusion
- **GIVEN** coverage contains a supported negative issue
- **AND** no blocking unknowns remain for that issue
- **WHEN** the final answer is submitted
- **THEN** the answer SHALL state the negative conclusion directly rather than falling back to an evidence summary
