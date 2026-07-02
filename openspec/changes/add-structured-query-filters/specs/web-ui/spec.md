## ADDED Requirements

### Requirement: Web API returns structured filter diagnostics

The system SHALL include structured filter diagnostics in `POST /api/query` JSON responses.

#### Scenario: API query parses filters

- **GIVEN** a JSON query request contains `效果是除外或者回收对手墓地的卡的四星超量怪兽`
- **WHEN** `POST /api/query` is called
- **THEN** the JSON response includes parsed structured filters
- **AND** includes the normalized effect query
- **AND** includes filter candidate diagnostics when available

#### Scenario: Structured bot response includes filters

- **GIVEN** a JSON query response includes the `structured` response object
- **AND** structured filters were parsed
- **WHEN** the response is serialized
- **THEN** the `structured` object includes parsed filter diagnostics outside individual card blocks

### Requirement: Web UI displays filter diagnostics

The system SHALL display structured filter diagnostics in the local Web UI when present.

#### Scenario: Browser query parses filters

- **GIVEN** the Web UI is open
- **AND** the user submits a query containing `四星超量怪兽`
- **WHEN** results are rendered
- **THEN** the page displays the parsed filters or filter warnings near the query results
