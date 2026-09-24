## ADDED Requirements

### Requirement: CLI exposes rift-query command

The system SHALL expose a `rift-query` command through `python -m rag_agent` that queries the Riftbound rules RAG through the gateway client and prints the answer, warnings, and cited rule ids.

#### Scenario: Successful rift-query

- **GIVEN** the upstream rift-rag service is reachable
- **WHEN** `python -m rag_agent rift-query "<question>"` runs
- **THEN** the printed output includes the upstream answer text
- **AND** the printed output includes each source rule id

#### Scenario: rift-query supports QA options

- **WHEN** `python -m rag_agent rift-query --help` is run
- **THEN** the help output includes `--top-k`, `--mode`, and `--trace`

#### Scenario: rift-query upstream failure

- **GIVEN** the upstream rift-rag service is unreachable
- **WHEN** `rift-query` runs
- **THEN** stderr begins with `error:`
- **AND** the command returns exit code `1`
