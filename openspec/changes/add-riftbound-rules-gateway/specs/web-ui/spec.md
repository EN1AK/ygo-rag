## ADDED Requirements

### Requirement: Web app exposes Riftbound rules QA endpoint

The system SHALL expose `POST /api/rift/query` that proxies Riftbound rules questions through the Riftbound rules gateway and returns the gateway response as JSON.

#### Scenario: Valid Riftbound QA request

- **GIVEN** a JSON body containing a non-empty `query` and optionally `top_k`, `mode`, and `trace`
- **WHEN** `POST /api/rift/query` is called
- **THEN** the request is validated and forwarded to the upstream rift-rag service
- **AND** the response status is `200` with the gateway response JSON

#### Scenario: Invalid Riftbound QA request

- **GIVEN** a JSON body with an empty `query`, an out-of-range `top_k`, or an invalid `mode`
- **WHEN** `POST /api/rift/query` is called
- **THEN** the response status is `400`
- **AND** the response body is a JSON error message

### Requirement: Web app maps Riftbound upstream failures to JSON error statuses

The system SHALL return deterministic JSON error statuses from `POST /api/rift/query` when the upstream rift-rag service fails, instead of a generic `500`.

#### Scenario: Upstream unreachable or error status

- **GIVEN** the upstream rift-rag service is unreachable or responds with a non-`200` status
- **WHEN** `POST /api/rift/query` is called with a valid body
- **THEN** the response status is `502`
- **AND** the response body is a JSON error message

#### Scenario: Upstream timeout

- **GIVEN** the upstream rift-rag service exceeds the configured gateway timeout
- **WHEN** `POST /api/rift/query` is called with a valid body
- **THEN** the response status is `504`
- **AND** the response body is a JSON error message
