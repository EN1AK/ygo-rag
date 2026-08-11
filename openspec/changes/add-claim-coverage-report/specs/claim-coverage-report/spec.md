## ADDED Requirements

### Requirement: Claim coverage report is structured
The system SHALL produce a structured Claim Coverage Report for adjudication coverage critique instead of relying on an LLM-provided sufficient boolean.

#### Scenario: Critique coverage produces report
- **GIVEN** an adjudication scenario with resolved cards and retrieved evidence
- **WHEN** the planner calls `critique_coverage`
- **THEN** the tool SHALL return a Claim Coverage Report
- **AND** the report SHALL include required claims, evidence checks, gaps, suggested queries, and claim dependencies

#### Scenario: Report schema is invalid
- **GIVEN** the coverage LLM returns malformed JSON or omits required report fields
- **WHEN** the report is parsed
- **THEN** the system SHALL reject the report
- **AND** the planner observation SHALL expose the validation error without marking coverage sufficient

### Requirement: Required claims cover questions and branches
The system SHALL generate required claims for each user question and each explicit hypothetical branch before final submission.

#### Scenario: Main question claims are generated
- **GIVEN** a scenario with one direct ruling question
- **WHEN** claim coverage is critiqued
- **THEN** the report SHALL include one or more required claims needed to answer that question

#### Scenario: Hypothetical branch claims are generated
- **GIVEN** a scenario whose question asks what happens if a condition is assumed
- **WHEN** claim coverage is critiqued
- **THEN** the report SHALL include separate required claims for the main case and the hypothetical branch
- **AND** the hypothetical branch claims SHALL be identifiable by claim type or branch metadata

### Requirement: Claim statuses are expressive
The system SHALL classify each claim using exactly one of `supported`, `partially_supported`, `unsupported`, `contradicted`, or `unknown`.

#### Scenario: Claim is supported
- **GIVEN** evidence that is relevant and entails the claim
- **WHEN** the claim is assessed
- **THEN** the claim status SHALL be `supported`

#### Scenario: Claim is partially supported
- **GIVEN** evidence that supports only a prerequisite, analogy, or incomplete part of the claim
- **WHEN** the claim is assessed
- **THEN** the claim status SHALL be `partially_supported`

#### Scenario: Claim is unsupported
- **GIVEN** relevant evidence that does not entail the claim
- **WHEN** the claim is assessed
- **THEN** the claim status SHALL be `unsupported`

#### Scenario: Claim is contradicted
- **GIVEN** evidence that supports the opposite of the claim
- **WHEN** the claim is assessed
- **THEN** the claim status SHALL be `contradicted`

#### Scenario: Claim is unknown
- **GIVEN** no relevant evidence is available for the claim
- **WHEN** the claim is assessed
- **THEN** the claim status SHALL be `unknown`

### Requirement: Evidence relevance and entailment are assessed separately
The system SHALL require each claim evidence check to distinguish whether evidence is relevant from whether it entails the claim.

#### Scenario: Evidence is relevant but not entailing
- **GIVEN** an FAQ about the same card or mechanism
- **AND** the FAQ does not prove the proposed claim
- **WHEN** the evidence is checked for a claim
- **THEN** the relevance assessment MAY be high
- **AND** the entailment assessment SHALL NOT be treated as supporting the claim

#### Scenario: Evidence contradicts a claim
- **GIVEN** an FAQ whose ruling implies the opposite of the claim
- **WHEN** the evidence is checked for that claim
- **THEN** the evidence check SHALL record contradictory entailment
- **AND** the claim status SHALL be `contradicted` unless stronger supporting evidence resolves the conflict

### Requirement: Claim dependency graph is exposed
The system SHALL expose dependencies between claims so final conclusions can be traced through prerequisite claims.

#### Scenario: Claim depends on card text and ruling principle
- **GIVEN** a case-application claim depends on a card-text claim and a rule-principle claim
- **WHEN** the Claim Coverage Report is produced
- **THEN** the case-application claim SHALL list the prerequisite claim ids in `depends_on`

#### Scenario: Dependency references unknown claim
- **GIVEN** a report dependency references a claim id not present in the report
- **WHEN** the report is validated
- **THEN** the report SHALL be rejected

### Requirement: Claim gaps drive targeted retrieval
The system SHALL expose claim-level gaps and suggested retrieval queries to guide subsequent planner tool calls.

#### Scenario: Required claim lacks support
- **GIVEN** a required claim has status `unsupported`, `partially_supported`, or `unknown`
- **WHEN** the Claim Coverage Report is returned
- **THEN** the report SHALL include a gap for that claim
- **AND** the gap SHOULD include one or more suggested FAQ or rules queries

#### Scenario: Planner receives gaps
- **GIVEN** the latest Claim Coverage Report contains gaps
- **WHEN** the planner prompt is built
- **THEN** the prompt SHALL include the claim gaps and suggested queries
- **AND** the planner MAY use them to choose a targeted retrieval tool

### Requirement: Submit gate is computed from claims
The system SHALL compute final coverage sufficiency from the Claim Coverage Report rather than trusting an LLM-provided sufficient flag.

#### Scenario: Required claims are supported
- **GIVEN** every decisive required claim is `supported`
- **AND** every cited dependency is also `supported`
- **WHEN** coverage is aggregated
- **THEN** final submission MAY be allowed

#### Scenario: Required claim is unsupported
- **GIVEN** any decisive required claim is `unsupported`, `contradicted`, or `unknown`
- **WHEN** coverage is aggregated
- **THEN** normal-confidence final submission SHALL be blocked
- **AND** the planner SHALL receive the claim gap or contradiction

#### Scenario: Required claim is partially supported
- **GIVEN** any decisive required claim is `partially_supported`
- **WHEN** coverage is aggregated
- **THEN** final submission SHALL be blocked unless the final answer explicitly reports low confidence or uncertainty for the affected conclusion
