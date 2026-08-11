## ADDED Requirements

### Requirement: LangGraph adjudication state
The system SHALL represent adjudication as a LangGraph state machine with explicit state for scenario text, parsed questions, card mentions, card candidates, resolved cards, evidence, coverage status, warnings, tool trace, loop counters, and final answer JSON.

#### Scenario: State is initialized from scenario text
- **WHEN** an adjudication agent run starts with a user scenario
- **THEN** the graph state contains the original scenario text, empty candidate/evidence collections, zeroed loop counters, and no final answer

#### Scenario: State records node outputs
- **WHEN** a graph node extracts cards, retrieves evidence, or checks coverage
- **THEN** the graph state records the node output and appends a trace entry with the node name and outcome summary

### Requirement: Fuzzy card resolution through card RAG
The system SHALL resolve card mentions using exact matching, normalized matching, and the existing card retrieval/RAG channel before final evidence retrieval.

#### Scenario: Middle-dot name resolves through normalization
- **WHEN** the user scenario mentions `破械冥官·篁` and the local card database stores `破械冥官 篁`
- **THEN** the card-resolution tool returns `破械冥官 篁` with card id `68625623` as a high-confidence candidate

#### Scenario: Ambiguous fuzzy match is not silently selected
- **WHEN** a mention produces multiple close card candidates above the ambiguity threshold
- **THEN** the graph marks the mention as ambiguous and includes all plausible candidates in state instead of silently choosing one

#### Scenario: Unresolved card mention triggers another search
- **WHEN** scenario extraction finds a likely card mention that is not resolved by exact or normalized matching
- **THEN** the graph calls the card retrieval/RAG search tool with a query derived from that mention and nearby scenario text

### Requirement: Agent tools for ruling evidence
The system SHALL expose deterministic graph tools for card effects, direct FAQ, related FAQ, analogous FAQ, supplement, and OCG rules retrieval.

#### Scenario: Direct and related FAQ tools label provenance
- **WHEN** FAQ evidence is retrieved by a primary card id match
- **THEN** the evidence relationship is labeled `direct_card`
- **AND** WHEN FAQ evidence is retrieved through `faqs[*].refer`
- **THEN** the evidence relationship is labeled `related_card`

#### Scenario: Analogous FAQ tool uses mechanism queries
- **WHEN** direct and related evidence does not cover a mechanism needed by a question
- **THEN** the graph calls analogous FAQ search with scenario terms, resolved card effect tags, and coverage-gap terms
- **AND** returned analogous evidence is labeled `analogous`

#### Scenario: Rules tool runs with targeted mechanism query
- **WHEN** the coverage checker identifies a general rules gap
- **THEN** the graph calls OCG rules search with a targeted mechanism query instead of only the original scenario text

### Requirement: Iterative coverage checking
The system SHALL evaluate whether each question and hypothetical branch has sufficient evidence before producing a final answer.

#### Scenario: Hypothetical branch requires evidence
- **WHEN** a user question contains an if, suppose, even-if, 若, 如果, or 假设 branch
- **THEN** the coverage checker requires evidence for both the actual case and the hypothetical branch before marking the answer coverage sufficient

#### Scenario: Missing hypothetical evidence triggers retrieval
- **WHEN** a hypothetical branch lacks evidence about a later effect-processing consequence
- **THEN** the graph performs another retrieval pass using gap terms such as target not destroyed, later processing, special summon, and not applied

#### Scenario: Retry limit reports gaps
- **WHEN** the graph reaches the configured maximum retrieval iterations without sufficient coverage
- **THEN** the graph proceeds to final output with coverage gaps listed in warnings or uncertainties

### Requirement: Evidence-grounded final JSON answer
The system SHALL produce a validated JSON answer from the final answer node using only evidence ids present in the graph state.

#### Scenario: Final answer contains fixed fields
- **WHEN** the graph reaches the final answer node
- **THEN** the returned answer JSON contains `conclusion`, `answers`, `evidence_use`, and `uncertainties`

#### Scenario: Each question has an answer object
- **WHEN** the parsed scenario contains multiple questions
- **THEN** the `answers` array contains one answer object per question

#### Scenario: Unknown evidence id is rejected
- **WHEN** the final answer cites an evidence id not present in graph state
- **THEN** the system rejects the answer and retries or returns a deterministic fallback with the validation error recorded

### Requirement: Agent trace is auditable
The system SHALL expose a compact tool-call trace for adjudication agent runs.

#### Scenario: Trace includes card search
- **WHEN** fuzzy card search is called during an agent run
- **THEN** the trace records the search query, candidate count, selected candidates, and ambiguity status

#### Scenario: Trace includes coverage decisions
- **WHEN** the coverage checker routes the graph to another retrieval node or to final answer
- **THEN** the trace records the coverage decision and the gap reasons used for routing

### Requirement: Existing adjudication compatibility
The system SHALL preserve existing adjudication entry points while adding the LangGraph agent path.

#### Scenario: Existing API shape remains usable
- **WHEN** a CLI or Web caller runs adjudication without requesting debug trace fields
- **THEN** the response still includes scenario, resolved cards, evidence, reasoning or answer, confidence, warnings, and simulation status fields compatible with the existing response shape

#### Scenario: Agent path can be selected
- **WHEN** the caller enables the LangGraph adjudication agent path through configuration, CLI flag, or service option
- **THEN** adjudication uses the graph loop and includes agent coverage and trace data when requested
