## ADDED Requirements

### Requirement: Retrieval ranking does not imply card resolution validity
The system SHALL distinguish retrieval score from card resolution validity.

#### Scenario: Retrieved card has high relative rank
- **GIVEN** sparse or dense retrieval returns a card candidate with the highest score
- **WHEN** card resolution consumes that candidate
- **THEN** the retrieval score SHALL be treated as ranking evidence only
- **AND** a separate validity decision SHALL determine whether the mention is resolved

### Requirement: Sparse card candidates expose source and validation inputs
The system SHALL expose enough candidate metadata for downstream validation.

#### Scenario: Candidate provider returns sparse candidate
- **WHEN** sparse card retrieval returns a candidate for card resolution
- **THEN** the candidate SHALL include card id, name, score, retrieval source, match reason, card text, and normalized-name comparison data needed for validation

### Requirement: Persistent ruling retrieval preserves stronger relationship labels
The system SHALL compute persistent ruling candidate relationship labels from document metadata and requested card ids before final ranking.

#### Scenario: Candidate has direct relationship
- **GIVEN** a persistent ruling candidate matches requested card ids through primary card metadata
- **WHEN** the candidate is ranked or reranked
- **THEN** its relationship SHALL preserve `direct_card`
- **AND** ranking and reranking SHALL NOT replace that label with `analogous`

#### Scenario: Candidate has related relationship
- **GIVEN** a persistent ruling candidate matches requested card ids through refer metadata
- **WHEN** the candidate is ranked or reranked
- **THEN** its relationship SHALL preserve `related_card`
- **AND** ranking and reranking SHALL NOT replace that label with `analogous`
