## ADDED Requirements

### Requirement: CLI query exposes structured filter diagnostics

The system SHALL include structured filter diagnostics in CLI query output when structured filters are parsed or filter warnings occur.

#### Scenario: CLI query parses filters

- **GIVEN** a valid database path
- **AND** the query contains `四星超量怪兽`
- **WHEN** `python -m rag_agent query <text> --db <path>` runs
- **THEN** the CLI output includes the parsed structured filters or a clearly labeled diagnostic section

#### Scenario: CLI query has no parsed filters

- **GIVEN** a valid database path
- **AND** the query contains no recognized structure constraints
- **WHEN** `python -m rag_agent query <text> --db <path>` runs
- **THEN** the CLI preserves the existing retrieval answer behavior
