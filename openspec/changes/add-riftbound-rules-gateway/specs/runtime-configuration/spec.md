## ADDED Requirements

### Requirement: Riftbound upstream endpoint is configurable

The system SHALL read the upstream rift-rag service base URL from `RIFTBOUND_RAG_BASE_URL` with a documented default.

#### Scenario: Upstream base URL not configured

- **GIVEN** `RIFTBOUND_RAG_BASE_URL` is not set
- **WHEN** settings are loaded
- **THEN** the Riftbound upstream base URL defaults to `http://127.0.0.1:7862`

#### Scenario: Upstream base URL overridden

- **GIVEN** `RIFTBOUND_RAG_BASE_URL` is set to a custom URL
- **WHEN** settings are loaded
- **THEN** the Riftbound gateway targets the configured URL

### Requirement: Riftbound upstream timeout is configurable

The system SHALL read the upstream request timeout from `RIFTBOUND_RAG_TIMEOUT_SECONDS` with a documented default, and SHALL fall back to the default when the value is missing or not a positive number.

#### Scenario: Upstream timeout not configured

- **GIVEN** `RIFTBOUND_RAG_TIMEOUT_SECONDS` is not set
- **WHEN** settings are loaded
- **THEN** the Riftbound upstream timeout defaults to `200` seconds

#### Scenario: Invalid upstream timeout value

- **GIVEN** `RIFTBOUND_RAG_TIMEOUT_SECONDS` is set to a non-numeric or non-positive value
- **WHEN** settings are loaded
- **THEN** the Riftbound upstream timeout falls back to `200` seconds
