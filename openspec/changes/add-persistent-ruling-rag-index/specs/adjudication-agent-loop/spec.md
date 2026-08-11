## ADDED Requirements

### Requirement: Agent evidence tools prefer persistent ruling indexes
The adjudication agent SHALL use persistent FAQ/supplement and OCG rules RAG indexes for evidence retrieval when those indexes are available and valid.

#### Scenario: Persistent FAQ index is available
- **GIVEN** the persistent FAQ/supplement index exists and passes manifest validation
- **WHEN** the adjudication agent retrieves direct, related, or analogous FAQ evidence
- **THEN** the agent uses the persistent ruling retrieval pipeline before runtime ygocdb parsing
- **AND** trace metadata records that persistent FAQ retrieval was used

#### Scenario: Persistent rules index is available
- **GIVEN** the persistent OCG rules index exists and passes manifest validation
- **WHEN** the adjudication agent retrieves general rules evidence
- **THEN** the agent uses the persistent rules retrieval pipeline before runtime EPUB parsing
- **AND** trace metadata records that persistent rules retrieval was used

### Requirement: Agent coverage uses reranked ruling evidence
The adjudication agent SHALL perform coverage checks over evidence returned by persistent ruling retrieval after hybrid recall and optional reranking.

#### Scenario: Hypothetical branch evidence found through persistent FAQ retrieval
- **GIVEN** a scenario contains a hypothetical branch about later effect processing
- **AND** persistent FAQ retrieval returns FAQ `ygocdb:faq:24345` or equivalent later-processing evidence
- **WHEN** coverage is checked
- **THEN** the hypothetical branch is considered covered by that evidence
- **AND** the final answer may cite the retrieved evidence id

#### Scenario: Persistent retrieval returns insufficient evidence
- **GIVEN** persistent ruling retrieval returns no evidence for a required branch
- **WHEN** coverage is checked
- **THEN** the agent records the evidence gap
- **AND** the agent may perform targeted retrieval using expanded mechanism terms
