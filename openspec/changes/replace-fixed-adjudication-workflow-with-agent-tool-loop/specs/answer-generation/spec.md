## ADDED Requirements

### Requirement: Final adjudication answers are submitted through a validated tool
The system SHALL accept final agent-loop adjudication answers only through `submit_ruling`.

#### Scenario: Submit ruling succeeds
- **GIVEN** the planner submits a strict JSON answer through `submit_ruling`
- **AND** every cited evidence id exists in the current evidence pool
- **WHEN** the answer passes structured response validation
- **THEN** the system SHALL accept the submitted answer as the final adjudication response

#### Scenario: Submit ruling cites unknown evidence
- **GIVEN** the planner submits an answer citing an evidence id not present in the current evidence pool
- **WHEN** `submit_ruling` validates the answer
- **THEN** the submission SHALL be rejected
- **AND** the validation error SHALL be visible to the next planner iteration

### Requirement: Submit ruling accounts for unresolved card state
The system SHALL require final submissions to account for unresolved or ambiguous card mentions.

#### Scenario: Unresolved cards remain
- **GIVEN** the work state contains unresolved or ambiguous card mentions
- **WHEN** the planner calls `submit_ruling`
- **THEN** the submitted answer SHALL either list those mentions as uncertainties or use a low-confidence conclusion
- **AND** the system SHALL reject submissions that present unresolved card-dependent conclusions as fully certain

### Requirement: Fallback answer is used when submission cannot be accepted
The system SHALL return a fallback answer over gathered evidence when the agent loop cannot produce a valid submission.

#### Scenario: Planner repeatedly submits invalid answers
- **GIVEN** the planner has produced invalid final submissions up to the configured retry limit
- **WHEN** no valid final submission is available
- **THEN** the system SHALL return a deterministic fallback answer over the gathered evidence
- **AND** the response SHALL include warnings describing the submission validation failures
