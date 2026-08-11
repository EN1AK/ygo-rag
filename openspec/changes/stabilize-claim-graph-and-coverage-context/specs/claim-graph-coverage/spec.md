## ADDED Requirements

### Requirement: Coverage prompts preserve essential evidence context
The system SHALL build a typed coverage evidence context instead of passing only the most recent evidence items.

#### Scenario: Resolved card effects are always included
- **GIVEN** resolved cards have `card_effect` evidence in agent state
- **AND** later retrieval has added FAQ, supplement, or rules evidence
- **WHEN** the planner or coverage critic prompt is built
- **THEN** every resolved card's `card_effect` evidence SHALL be included in the prompt context
- **AND** the card effect text SHALL NOT be omitted due to recency truncation

#### Scenario: Claim-cited evidence is retained
- **GIVEN** a current claim graph cites evidence ids in evidence checks
- **WHEN** a later coverage critic prompt is built
- **THEN** cited evidence SHALL be included or summarized in the prompt context
- **AND** the prompt SHALL identify which claims cite that evidence

### Requirement: Claim graph identities are stable
The system SHALL preserve claim or issue identities across repeated coverage critiques.

#### Scenario: Subsequent critique updates existing claims
- **GIVEN** a claim graph already exists
- **AND** new evidence has been retrieved
- **WHEN** `critique_coverage` runs again
- **THEN** existing claim or issue ids SHALL retain their original meanings
- **AND** the critique SHALL update statuses, evidence checks, gaps, and suggested queries instead of renumbering equivalent claims

#### Scenario: New required claim is discovered
- **GIVEN** a later critique identifies a required claim not present in the current graph
- **WHEN** the claim is accepted
- **THEN** the system SHALL append it with a new stable id
- **AND** existing ids SHALL NOT be reused for the new claim

### Requirement: Coverage represents conclusion polarity
The system SHALL distinguish affirmative, negative, conditional, and unknown conclusions for adjudication issues.

#### Scenario: Evidence supports a negative answer
- **GIVEN** evidence contradicts an affirmative proposition
- **AND** the same evidence entails the negative answer to an adjudication issue
- **WHEN** coverage is aggregated
- **THEN** the issue SHALL be allowed to become supported with negative polarity
- **AND** final submission SHALL be allowed to cite that supported negative issue

#### Scenario: Evidence is merely absent
- **GIVEN** no evidence supports either the affirmative or negative answer for an issue
- **WHEN** coverage is aggregated
- **THEN** the issue SHALL remain unknown or unsupported
- **AND** the system SHALL NOT treat the absence of support as support for the opposite answer

### Requirement: Case application may be dependency-supported
The system SHALL allow case-application conclusions to be supported by explicit dependency inference.

#### Scenario: Application is supported by parent claims
- **GIVEN** a `case_application` claim depends on supported `card_text` and `rule_principle` or `ruling_fact` claims
- **AND** the application includes an inference reason
- **AND** no cited evidence contradicts the application
- **WHEN** coverage is aggregated
- **THEN** the application MAY be considered supported without an exact same-scenario FAQ

#### Scenario: Application lacks inference basis
- **GIVEN** a `case_application` claim has no direct entailing evidence
- **AND** it does not list supported dependencies with an inference reason
- **WHEN** coverage is aggregated
- **THEN** the application SHALL remain insufficient for confident submission

### Requirement: Contradictions are actionable
The system SHALL expose contradictions as answerable negative or conditional outcomes when evidence supports them.

#### Scenario: Positive claim is contradicted
- **GIVEN** a positive claim says an effect can apply
- **AND** evidence states that the effect cannot apply in the relevant situation
- **WHEN** coverage critique processes the evidence
- **THEN** the positive claim SHALL be marked contradicted or superseded
- **AND** the corresponding issue SHALL record a supported negative answer when the contradiction entails the issue outcome

### Requirement: Coverage response remains traceable
The system SHALL serialize stable claim or issue graph data in agent coverage and trace output.

#### Scenario: Agent response includes coverage graph
- **GIVEN** the agent has run at least one successful coverage critique
- **WHEN** the adjudication response is serialized
- **THEN** `agent_coverage` SHALL include the latest stable claim or issue graph
- **AND** it SHALL include aggregation status, blocking ids, gaps, and suggested queries

### Requirement: Critique repetition is evidence-aware
The system SHALL not treat coverage critique after new evidence as an ordinary duplicate tool call.

#### Scenario: New evidence then critique
- **GIVEN** `critique_coverage` has already run
- **AND** a retrieval tool adds new evidence
- **WHEN** the planner calls `critique_coverage` again
- **THEN** the call SHALL NOT be marked as duplicate
- **AND** it SHALL be recorded as a new evidence-revision review

#### Scenario: Same evidence repeated critique
- **GIVEN** no state or evidence has changed since the previous critique
- **WHEN** the planner calls `critique_coverage` again
- **THEN** the system MAY warn about an unproductive repeated critique

### Requirement: Loop budgets are separated
The system SHALL track retrieval, critique, submit, and total planner budgets separately.

#### Scenario: Critique does not consume retrieval budget
- **GIVEN** the agent runs multiple coverage critiques between retrieval calls
- **WHEN** retrieval budget is checked
- **THEN** critique calls SHALL NOT reduce the remaining retrieval call budget

#### Scenario: Budget exhaustion is reported accurately
- **GIVEN** the agent ends without accepted `submit_ruling` because a budget limit was reached
- **WHEN** the final response is built
- **THEN** warnings SHALL identify the exhausted budget category
- **AND** the response SHALL NOT describe the cause only as invalid LLM synthesis
