## ADDED Requirements

### Requirement: CLI can translate arbitrary text

The system SHALL expose a `translate` command through `python -m rag_agent` for invoking the translation service from the command line.

#### Scenario: User asks for top-level help
- **WHEN** `python -m rag_agent --help` is run
- **THEN** the help output lists `translate`

#### Scenario: Translate with defaults
- **GIVEN** DeepSeek credentials are configured
- **WHEN** `python -m rag_agent translate "hello"` is run
- **THEN** the command translates the text to Chinese by default
- **AND** prints the translated text

#### Scenario: Translate with explicit target language
- **GIVEN** DeepSeek credentials are configured
- **WHEN** `python -m rag_agent translate "你好" --target-lang en` is run
- **THEN** the command translates the text to English

### Requirement: CLI translation validates input

The system SHALL reject invalid translation CLI input with the existing CLI error handling behavior.

#### Scenario: Empty CLI translation text
- **GIVEN** an empty or whitespace-only translation string
- **WHEN** the `translate` command runs
- **THEN** stderr begins with `error:`
- **AND** the command returns exit code `1`
