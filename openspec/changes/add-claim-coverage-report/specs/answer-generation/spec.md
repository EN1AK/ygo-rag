## ADDED Requirements

### Requirement: Adjudication answers cite supported claims
The system SHALL require agent-loop adjudication final answers to cite claim ids from the latest accepted Claim Coverage Report in addition to evidence ids.

#### Scenario: Final answer cites supported claim
- **GIVEN** a final adjudication answer submitted through `submit_ruling`
- **AND** the answer cites a claim id whose status is `supported`
- **AND** the claim cites valid evidence ids
- **WHEN** the answer is validated
- **THEN** the cited claim SHALL be accepted as support for the answer judgment

#### Scenario: Final answer cites unknown claim id
- **GIVEN** a final adjudication answer cites a claim id not present in the latest Claim Coverage Report
- **WHEN** the answer is validated
- **THEN** the submission SHALL be rejected
- **AND** the validation error SHALL identify the unknown claim id

### Requirement: Unsupported claims block confident answers
The system SHALL reject or downgrade final answers that rely on unsupported, contradicted, unknown, or partially supported decisive claims.

#### Scenario: Answer relies on unsupported claim
- **GIVEN** a final answer cites a decisive claim with status `unsupported`
- **WHEN** `submit_ruling` validates the answer
- **THEN** the submission SHALL be rejected unless the answer explicitly presents the conclusion as uncertain or low confidence

#### Scenario: Answer relies on contradicted claim
- **GIVEN** a final answer cites a decisive claim with status `contradicted`
- **WHEN** `submit_ruling` validates the answer
- **THEN** the submission SHALL be rejected
- **AND** the validation error SHALL identify the contradictory claim

#### Scenario: Answer relies on unknown claim
- **GIVEN** a final answer cites a decisive claim with status `unknown`
- **WHEN** `submit_ruling` validates the answer
- **THEN** the submission SHALL be rejected unless the answer explicitly states that the relevant conclusion cannot be determined from available evidence

#### Scenario: Answer relies on partially supported claim
- **GIVEN** a final answer cites a decisive claim with status `partially_supported`
- **WHEN** `submit_ruling` validates the answer
- **THEN** the submission SHALL be rejected unless the answer identifies the missing support and uses low confidence or uncertainty for the affected conclusion

### Requirement: Hypothetical branch answers cite branch claims
The system SHALL require answers to hypothetical branches to cite claims that are explicitly associated with that branch.

#### Scenario: Hypothetical branch has supported claim
- **GIVEN** a user question contains a hypothetical branch
- **AND** the final answer includes `hypothetical_conclusion`
- **WHEN** the answer is validated
- **THEN** the hypothetical conclusion SHALL cite at least one claim associated with the hypothetical branch

#### Scenario: Hypothetical branch lacks branch claim
- **GIVEN** a user question contains a hypothetical branch
- **AND** the final answer includes a certain hypothetical conclusion
- **AND** no cited supported claim is associated with the hypothetical branch
- **WHEN** the answer is validated
- **THEN** the submission SHALL be rejected

### Requirement: Evidence relationship claims remain auditable
The system SHALL preserve evidence relationship labels used by claims and final answers.

#### Scenario: Answer misstates evidence relationship
- **GIVEN** a final answer's evidence use says an evidence id is `direct_card`
- **AND** the evidence pool records that evidence only as `analogous`
- **WHEN** the answer is validated
- **THEN** the submission SHALL be rejected or corrected before acceptance

#### Scenario: Claim uses analogous evidence for decisive conclusion
- **GIVEN** a decisive claim is supported only by analogous evidence
- **WHEN** final answer validation runs
- **THEN** the answer SHALL either cite the analogy limitation in uncertainties or be rejected as overconfident
