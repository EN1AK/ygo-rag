## ADDED Requirements

### Requirement: CLI exposes adjudication command

The system SHALL expose a command-line entry point for AI ruling adjudication without changing the existing `query` command behavior.

#### Scenario: User asks for CLI help

- **WHEN** `python -m rag_agent --help` is run
- **THEN** the help output lists an adjudication command or mode
- **AND** the existing query command remains listed

#### Scenario: User runs adjudication from CLI

- **GIVEN** a valid card database path
- **WHEN** the user runs the adjudication command with a scenario text
- **THEN** the system executes the adjudication workflow
- **AND** prints a Chinese adjudication answer

### Requirement: CLI adjudication accepts evidence source options

The system SHALL allow CLI adjudication callers to configure local ygocdb cache/index and OCG rules EPUB/index evidence sources.

#### Scenario: User provides ygocdb and rules sources

- **GIVEN** valid local paths for the card database, ygocdb cache or index, and OCG rules EPUB or index
- **WHEN** adjudication runs from the CLI
- **THEN** the system uses those paths for evidence gathering

#### Scenario: Optional evidence source is missing

- **GIVEN** a card database path is valid
- **AND** no ygocdb cache/index or OCG rules EPUB/index is provided
- **WHEN** adjudication runs from the CLI
- **THEN** the command still runs with warnings for unavailable optional evidence sources

### Requirement: CLI adjudication can request simulation

The system SHALL expose a CLI option to request local simulator verification when the simulator adapter is available.

#### Scenario: Simulation flag is absent

- **WHEN** adjudication runs without the simulation option
- **THEN** the response indicates that simulation was not requested

#### Scenario: Simulation flag is present but unavailable

- **WHEN** adjudication runs with the simulation option
- **AND** the simulator adapter is unavailable
- **THEN** the command returns a normal adjudication response with a simulator warning
