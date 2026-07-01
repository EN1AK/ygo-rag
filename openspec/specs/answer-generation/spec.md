# answer-generation Specification

## Purpose

Define the current behavior for formatting retrieval results and optionally generating a Chinese answer with DeepSeek.

## Requirements

### Requirement: Retrieval-only answers include query and candidate details

The system SHALL format retrieval-only answers as Chinese text containing the original query and ranked candidate cards.

#### Scenario: Candidates are available

- **GIVEN** retrieved card candidates
- **WHEN** no LLM is configured for the query
- **THEN** the answer includes the query
- **AND** each candidate includes rank, card name, card id, score, original source text, and a retrieval reason

#### Scenario: No candidates are available

- **GIVEN** no retrieved card candidates
- **WHEN** no LLM is configured for the query
- **THEN** the answer states that no sufficiently similar candidate cards were found

### Requirement: Retrieval reasons are source-based

The system SHALL provide a fixed reason string based on the candidate source.

#### Scenario: Reranked candidate

- **GIVEN** a candidate source is `reranker`
- **WHEN** it is converted to a retrieved card
- **THEN** the reason states that hybrid candidates were reranked by local `bge-reranker-v2-m3`

#### Scenario: Dense candidate

- **GIVEN** a candidate source is `dense`
- **WHEN** it is converted to a retrieved card
- **THEN** the reason states that dense vector search matched semantic effect text

### Requirement: DeepSeek answer generation is optional

The system SHALL call DeepSeek only when LLM answer generation is requested.

#### Scenario: LLM disabled

- **GIVEN** query execution does not request LLM
- **WHEN** the answer is produced
- **THEN** the system returns retrieval-only formatted text

#### Scenario: LLM enabled

- **GIVEN** query execution requests LLM
- **AND** `DEEPSEEK_API_KEY` is available
- **WHEN** the answer is produced
- **THEN** the system sends a prompt containing the user query and retrieved candidate card context to the configured DeepSeek-compatible chat model

### Requirement: LLM prompt is constrained to retrieved context

The system SHALL instruct the LLM to answer only from provided candidate card text and not invent unprovided cards or effects.

#### Scenario: Build answer prompt

- **GIVEN** retrieved candidate cards
- **WHEN** the LLM prompt is built
- **THEN** it includes candidate card names, ids, retrieval scores, original text, and retrieval reasons
- **AND** it asks for a Chinese answer listing similar cards and explaining whether similarities are only keyword-level

### Requirement: LLM judge prompt is separate from final answer prompt

The system SHALL use a dedicated prompt for LLM rerank that is separate from the final answer synthesis prompt.

#### Scenario: LLM rerank without final LLM answer

- **GIVEN** LLM rerank is enabled
- **AND** final LLM answer generation is disabled
- **WHEN** a query is executed
- **THEN** the system uses the LLM judge prompt for ranking
- **AND** returns retrieval-only answer formatting for the final answer text

#### Scenario: LLM rerank with final LLM answer

- **GIVEN** LLM rerank is enabled
- **AND** final LLM answer generation is enabled
- **WHEN** a query is executed
- **THEN** the system first uses the LLM judge prompt for ranking
- **AND** then uses the final answer prompt with the reranked candidates

### Requirement: LLM judge output is structured

The system SHALL require LLM judge output to be parseable structured data containing card ids, scores, and reasons.

#### Scenario: Valid judge output

- **GIVEN** the LLM judge returns structured data for candidate cards
- **WHEN** the response is parsed
- **THEN** each accepted scored item includes card id, numeric score, and a concise reason

#### Scenario: Invalid judge output

- **GIVEN** the LLM judge response cannot be parsed as the required structure
- **WHEN** the response is processed
- **THEN** the system reports a clear LLM rerank parsing error

### Requirement: LLM judge reasons are exposed in candidate results

The system SHALL expose LLM judge reasons for candidates ranked by LLM rerank.

#### Scenario: Candidate ranked by LLM judge

- **GIVEN** LLM rerank assigns a reason to a candidate
- **WHEN** the candidate is returned to CLI or Web output
- **THEN** the candidate reason indicates LLM judge reranking
- **AND** includes or references the judge reason

## Open Questions

- Should LLM output be a structured schema instead of free-form text?
- Should every LLM claim be linked to a card id and source text quote?
- Should LLM generation be allowed to reorder retrieved candidates?
