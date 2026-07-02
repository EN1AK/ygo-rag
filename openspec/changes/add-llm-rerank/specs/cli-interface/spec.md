## ADDED Requirements

### Requirement: Query command can select LLM rerank

The system SHALL expose a CLI option that selects LLM rerank for query execution.

#### Scenario: Query help lists LLM rerank option

- **WHEN** `python -m rag_agent query --help` is run
- **THEN** the help output includes an option for selecting LLM rerank

#### Scenario: User selects LLM rerank

- **GIVEN** a valid database path
- **AND** DeepSeek credentials are configured
- **WHEN** the user runs `query <text>` with the LLM rerank option
- **THEN** query execution uses LLM rerank for candidate ordering

### Requirement: Existing local rerank flag remains compatible

The system SHALL preserve the existing `--rerank` behavior as local bge rerank.

#### Scenario: User runs existing rerank flag

- **GIVEN** a valid database path
- **WHEN** the user runs `query <text> --rerank`
- **THEN** query execution uses the existing local rerank behavior

### Requirement: LLM rerank and final LLM answer are independently selectable

The system SHALL allow LLM rerank to be selected independently from final LLM answer generation.

#### Scenario: LLM rerank without final LLM answer

- **GIVEN** DeepSeek credentials are configured
- **WHEN** the user enables LLM rerank but does not enable `--llm`
- **THEN** the system uses LLM rerank for ordering
- **AND** returns retrieval-only final answer formatting

#### Scenario: LLM rerank with final LLM answer

- **GIVEN** DeepSeek credentials are configured
- **WHEN** the user enables both LLM rerank and `--llm`
- **THEN** the system uses LLM rerank for ordering
- **AND** uses DeepSeek answer synthesis for the final answer
