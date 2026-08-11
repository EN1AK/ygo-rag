## ADDED Requirements

### Requirement: Ruling evidence documents are normalized for indexing
The system SHALL convert ygocdb FAQ entries, ygocdb supplement entries, and OCG rules passages into normalized retrieval documents with stable evidence ids and provenance metadata.

#### Scenario: FAQ document preserves provenance
- **GIVEN** a ygocdb card payload contains a FAQ entry with `fid`, `question`, `answer`, and `refer`
- **WHEN** ruling documents are built from the payload
- **THEN** the FAQ document evidence id is `ygocdb:faq:{fid}`
- **AND** the document metadata includes primary card ids, refer card ids, source URL, snapshot timestamp, and kind `faq`

#### Scenario: Supplement document preserves provenance
- **GIVEN** a ygocdb card payload contains a `supplement`
- **WHEN** ruling documents are built from the payload
- **THEN** the supplement document evidence id is `ygocdb:supplement:{card_id}`
- **AND** the document metadata includes primary card id, refer card ids, source URL, snapshot timestamp, and kind `card_supplement`

#### Scenario: Rules document preserves anchor metadata
- **GIVEN** an OCG rules EPUB is parsed into rule passages
- **WHEN** ruling documents are built from the passages
- **THEN** each rules document evidence id matches the parsed rule passage id
- **AND** metadata includes chapter path, XHTML path, anchor, title, and kind `rule_passage`

### Requirement: Persistent ruling indexes can be built from local sources
The system SHALL build persistent vector indexes for ruling FAQ/supplement documents and OCG rules documents from local cache files.

#### Scenario: Build FAQ index from local ygocdb cache
- **GIVEN** `data/ygocdb` contains per-card ygocdb API JSON payloads
- **AND** a valid cards database is available for card names and effect tags
- **WHEN** the FAQ index build command runs
- **THEN** FAQ and supplement documents are embedded into the configured persistent vector collection
- **AND** the build prints source counts, indexed document counts, and progress
- **AND** the build does not fetch network data

#### Scenario: Build rules index from local EPUB
- **GIVEN** the configured OCG rules EPUB exists
- **WHEN** the rules index build command runs
- **THEN** parsed rule passage documents are embedded into the configured persistent vector collection
- **AND** the build prints parsed passage counts, indexed document counts, and progress

### Requirement: Ruling index manifests are written and validated
The system SHALL write a manifest for each persistent ruling index and validate it before using dense retrieval.

#### Scenario: Manifest is written after build
- **WHEN** a ruling index build completes
- **THEN** a manifest records schema version, source paths, source counts, document count, embedding model, collection name, and build timestamp

#### Scenario: Stale or incompatible manifest disables dense retrieval
- **GIVEN** a persistent ruling index manifest is missing, stale, or incompatible with current settings
- **WHEN** ruling retrieval runs
- **THEN** dense retrieval for that index is skipped
- **AND** the caller receives a warning explaining why the persistent index was not used

### Requirement: Persistent ruling retrieval uses hybrid recall
The system SHALL retrieve FAQ, supplement, and rules evidence using sparse recall, optional dense vector recall, and reciprocal-rank fusion.

#### Scenario: Hybrid retrieval merges sparse and dense candidates
- **GIVEN** sparse and dense ruling retrieval both return candidates for a query
- **WHEN** hybrid ruling retrieval runs
- **THEN** candidates are deduplicated by evidence id
- **AND** reciprocal-rank fusion combines contributions from each ranked list
- **AND** final ordering is deterministic for ties

#### Scenario: Direct card filter boosts direct FAQ
- **GIVEN** the retrieval request includes resolved card ids
- **WHEN** FAQ retrieval runs
- **THEN** FAQ documents whose primary card ids include those card ids are available as direct-card evidence
- **AND** FAQ documents whose refer card ids include those card ids are available as related-card evidence

#### Scenario: Mechanism query retrieves analogous FAQ
- **GIVEN** the retrieval request includes scenario terms, effect tags, and mechanism terms
- **WHEN** analogous FAQ retrieval runs
- **THEN** FAQ candidates may be returned even when they do not directly reference the resolved card ids
- **AND** those candidates are labeled as analogous evidence

### Requirement: Ruling retrieval can be reranked
The system SHALL support optional local rerank and optional LLM judge rerank for ruling evidence candidates.

#### Scenario: Local rerank sorts ruling candidates
- **GIVEN** ruling evidence candidates have been fused
- **AND** local rerank is selected
- **WHEN** reranking runs
- **THEN** candidate snippets are passed to the configured local reranker
- **AND** final candidates are sorted by reranker score with deterministic tie breaking

#### Scenario: LLM judge rerank is candidate-bounded
- **GIVEN** ruling evidence candidates have been fused
- **AND** LLM judge rerank is selected
- **WHEN** the LLM rerank prompt is built
- **THEN** the prompt includes only retrieved candidate evidence ids, titles, snippets, relationships, sources, and retrieval scores
- **AND** the prompt instructs the LLM not to introduce outside evidence ids

#### Scenario: LLM judge unknown ids are rejected
- **GIVEN** LLM judge rerank receives candidate evidence ids `a` and `b`
- **WHEN** the LLM output ranks unknown evidence id `c`
- **THEN** evidence id `c` is ignored
- **AND** final ordering contains only retrieved candidates

#### Scenario: LLM judge failure falls back
- **GIVEN** LLM judge rerank is selected
- **WHEN** the LLM call times out or returns malformed output
- **THEN** the original fused or locally reranked candidate order is returned
- **AND** a warning records the rerank failure

### Requirement: Persistent ruling retrieval returns citation-ready evidence
The system SHALL convert retrieved ruling documents back into `EvidenceItem`-compatible records for adjudication and final-answer citation validation.

#### Scenario: Retrieved FAQ becomes evidence item
- **WHEN** a FAQ document is retrieved for adjudication
- **THEN** the evidence item uses the document evidence id, title, text, source, kind, card ids, relationship label, and metadata

#### Scenario: Retrieved rules passage becomes evidence item
- **WHEN** an OCG rules document is retrieved for adjudication
- **THEN** the evidence item uses the rule passage id, title, text, source `ocgrules`, relationship `rule_document`, and anchor metadata

### Requirement: Runtime fallback remains available
The system SHALL fall back to runtime ygocdb cache parsing and EPUB parsing when persistent ruling indexes are unavailable.

#### Scenario: FAQ index missing
- **GIVEN** a local ygocdb cache exists
- **AND** the persistent FAQ index is missing
- **WHEN** adjudication evidence retrieval runs
- **THEN** the system may build the existing runtime FAQ index
- **AND** a trace entry or warning indicates persistent FAQ retrieval was not used

#### Scenario: Rules index missing
- **GIVEN** a local rules EPUB exists
- **AND** the persistent rules index is missing
- **WHEN** adjudication evidence retrieval runs
- **THEN** the system may parse the EPUB at runtime
- **AND** a trace entry or warning indicates persistent rules retrieval was not used
