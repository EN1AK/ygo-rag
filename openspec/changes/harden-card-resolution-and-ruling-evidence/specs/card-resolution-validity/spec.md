## ADDED Requirements

### Requirement: Card resolution separates retrieval from validation
The system SHALL treat sparse and dense card retrieval as candidate generation, not as proof that a mention is resolved.

#### Scenario: Sparse candidate ranks first without validation
- **GIVEN** a card mention does not exactly match a card name or verified alias
- **AND** sparse retrieval returns a top-ranked candidate above the retrieval score threshold
- **WHEN** card resolution evaluates the candidate
- **THEN** the mention MUST NOT be marked `resolved` solely because that candidate ranks first
- **AND** the candidate MUST be marked `ambiguous`, `unresolved`, or `needs_validation`

#### Scenario: Exact normalized name match
- **GIVEN** a card mention normalizes to the same value as a local card name
- **WHEN** card resolution evaluates the mention
- **THEN** the mention SHALL be marked `resolved`
- **AND** the selection reason SHALL identify an exact normalized name match

### Requirement: Card resolution validates candidate compatibility
The system SHALL require a candidate card to prove lexical, alias, or constrained semantic compatibility with the original mention before selecting it.

#### Scenario: Candidate cannot explain mention
- **GIVEN** a retrieved candidate does not match the mention by exact name, verified alias, accepted substring coverage, or validated LLM explanation
- **WHEN** card resolution evaluates the candidate
- **THEN** the candidate MUST NOT be selected as the resolved card
- **AND** the trace SHALL include a rejection reason

#### Scenario: Candidate passes substring coverage
- **GIVEN** a mention is a substring of a candidate card name
- **AND** the mention satisfies configured minimum length and coverage constraints
- **WHEN** card resolution evaluates the candidate
- **THEN** the candidate MAY be selected without LLM validation
- **AND** the selection reason SHALL identify the substring coverage rule

### Requirement: Short mentions require strict proof
The system SHALL apply stricter validation for short card mentions and common abbreviations.

#### Scenario: Short mention has no verified alias
- **GIVEN** a card mention is shorter than the configured short-mention threshold
- **AND** the mention does not exactly match a card name or verified alias
- **WHEN** sparse or dense retrieval returns candidates
- **THEN** the mention MUST remain unresolved or ambiguous
- **AND** no candidate SHALL be selected solely from retrieval rank

#### Scenario: Short mention has verified alias
- **GIVEN** a card mention matches a verified local alias
- **WHEN** card resolution evaluates the mention
- **THEN** the mention SHALL resolve to the alias card id
- **AND** the selection reason SHALL include the alias source

### Requirement: LLM card selection is constrained and explanatory
The system SHALL accept LLM-assisted card selection only when the selected id is in the local candidate set and the output explains how the candidate matches the mention.

#### Scenario: LLM selects unknown id
- **GIVEN** the local candidate set contains card ids `1` and `2`
- **WHEN** the LLM returns selected card id `999`
- **THEN** the selected id MUST be rejected
- **AND** the mention SHALL remain ambiguous or unresolved

#### Scenario: LLM gives no compatibility reason
- **GIVEN** LLM card selection is enabled
- **AND** the LLM returns a candidate id from the local candidate set
- **WHEN** the LLM output does not explain how the candidate name or alias matches the mention
- **THEN** the selection MUST be rejected or downgraded to ambiguous
- **AND** the trace SHALL record the missing explanation

### Requirement: Card resolution trace exposes validation state
The system SHALL expose validation decisions in the card-resolution trace.

#### Scenario: Resolution completes
- **WHEN** card resolution finishes for a mention
- **THEN** the trace SHALL include candidate ids, candidate sources, validation status, validation reason, selection source, and unresolved or ambiguous reason
