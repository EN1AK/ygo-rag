## Purpose

Proxy Riftbound (符文战场) rules questions from local HTTP clients to the riftsim `rift-rag` service so that ygo-rag acts as the single local RAG entry point for the kami-man bot.

## ADDED Requirements

### Requirement: Gateway forwards validated QA requests to the upstream rift-rag service

The system SHALL forward a validated Riftbound QA request as JSON `POST {upstream_base_url}/api/query` with a body containing `query` (non-empty trimmed string), `top_k` (default 6), and `mode`/`trace` only when the caller provided them.

#### Scenario: Minimal request forwarded with defaults

- **GIVEN** a caller request `{"query": "反制堆叠有上限吗"}`
- **WHEN** the gateway processes the request
- **THEN** the upstream receives `query` equal to the trimmed question
- **AND** the upstream receives `top_k` equal to `6`
- **AND** no `mode` or `trace` key is sent upstream

#### Scenario: Explicit options forwarded verbatim

- **GIVEN** a caller request with `top_k` of `8`, `mode` of `oneshot`, and `trace` of `true`
- **WHEN** the gateway processes the request
- **THEN** the upstream receives those values unchanged

### Requirement: Gateway validates requests before calling upstream

The system SHALL reject invalid caller requests with a validation error and MUST NOT call the upstream service for rejected requests.

#### Scenario: Empty question rejected

- **GIVEN** a request whose `query` is missing, not a string, or whitespace-only
- **WHEN** the gateway processes the request
- **THEN** the system raises a validation error
- **AND** the upstream service is not contacted

#### Scenario: top_k out of range rejected

- **GIVEN** a request with `top_k` that is not an integer or is outside `1..20`
- **WHEN** the gateway processes the request
- **THEN** the system raises a validation error
- **AND** the upstream service is not contacted

#### Scenario: Invalid mode rejected

- **GIVEN** a request with `mode` other than `agent` or `oneshot`
- **WHEN** the gateway processes the request
- **THEN** the system raises a validation error
- **AND** the upstream service is not contacted

### Requirement: Gateway preserves the upstream response contract

The system SHALL return a success response containing string `answer`, list `warnings`, and list `sources` whose entries each carry `rule_id` and `topic`, and SHALL pass through the optional additive v2 fields `resolved_cards`, `coverage`, `exhausted`, `mode`, and `trace` unmodified whenever the upstream returns them.

#### Scenario: Legacy contract response passes through

- **GIVEN** an upstream `200` response containing only `answer`, `warnings`, and `sources`
- **WHEN** the gateway builds its response
- **THEN** the response contains `answer`, `warnings`, and `sources`
- **AND** no v2 field is synthesized

#### Scenario: v2 fields pass through unmodified

- **GIVEN** an upstream `200` response that also contains `resolved_cards`, `coverage`, `exhausted`, `mode`, or `trace`
- **WHEN** the gateway builds its response
- **THEN** those fields appear in the gateway response with values equal to the upstream values

### Requirement: Gateway maps upstream failures to deterministic errors

The system SHALL map every upstream failure to a deterministic gateway error category and MUST NOT leak a generic internal error for an upstream failure.

#### Scenario: Upstream unreachable

- **GIVEN** the upstream service is not running or refuses the connection
- **WHEN** the gateway processes a valid request
- **THEN** the system raises an upstream-unavailable error distinct from a validation error

#### Scenario: Upstream timeout

- **GIVEN** the upstream service accepts the connection but does not respond within the configured timeout
- **WHEN** the gateway processes a valid request
- **THEN** the system raises an upstream-timeout error distinct from other failure categories

#### Scenario: Upstream returns an error status

- **GIVEN** the upstream responds with a non-`200` status and a JSON `error` message
- **WHEN** the gateway processes a valid request
- **THEN** the system raises an upstream-failure error that includes the upstream error message

#### Scenario: Upstream success body is malformed

- **GIVEN** an upstream `200` response whose body lacks a string `answer` or a list `sources`
- **WHEN** the gateway builds its response
- **THEN** the system raises an upstream-failure error describing the malformed response
