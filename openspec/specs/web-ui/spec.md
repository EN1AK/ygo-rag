# web-ui Specification

## Purpose

Define the current behavior of the local Web UI and JSON query endpoint.

## Requirements

### Requirement: Web app serves a local query page

The system SHALL serve an HTML query page at `/`.

#### Scenario: Request index page

- **WHEN** a user sends `GET /`
- **THEN** the response is HTML with UTF-8 content type
- **AND** the page contains the `YGO RAG` title
- **AND** the page contains controls for query text, database path, top-k, rerank candidates, semantic retrieval, reranking, and LLM generation

### Requirement: Web app exposes JSON query API

The system SHALL expose `POST /api/query` for query execution.

#### Scenario: Valid query request

- **GIVEN** a JSON object containing a non-empty `query`
- **WHEN** `POST /api/query` is called
- **THEN** the system parses request options into a query request
- **AND** executes the configured query handler
- **AND** returns JSON containing `answer`, `results`, and `warnings`

### Requirement: Web API validates query parameters

The system SHALL validate query text, `top_k`, and `rerank_candidates`.

#### Scenario: Empty query

- **GIVEN** a request body with an empty or whitespace-only query
- **WHEN** `POST /api/query` is called
- **THEN** the response status is `400`
- **AND** the response includes an error message that query is required

#### Scenario: Numeric option out of range

- **GIVEN** `top_k` is outside `1..50` or `rerank_candidates` is outside `1..200`
- **WHEN** `POST /api/query` is called
- **THEN** the response status is `400`
- **AND** the response includes a validation error message

### Requirement: Web UI keeps credentials outside the page

The system SHALL not provide a browser field for DeepSeek API keys.

#### Scenario: LLM option is enabled

- **GIVEN** the user enables LLM generation in the browser
- **WHEN** the query is submitted
- **THEN** the server still reads DeepSeek credentials from runtime environment settings

### Requirement: Unsupported routes return JSON 404

The system SHALL return a JSON `404` response for unsupported HTTP routes.

#### Scenario: Unknown route

- **WHEN** a request is sent to an unsupported path
- **THEN** the response status is `404`
- **AND** the response body contains a JSON error

### Requirement: Application lifespan events are handled

The system SHALL handle ASGI lifespan startup and shutdown events.

#### Scenario: ASGI server starts and stops

- **WHEN** the ASGI server sends startup and shutdown lifespan events
- **THEN** the app responds with startup and shutdown completion messages

## Open Questions

- Should long-running queries be executed in a worker thread or background task?
- Should the Web API hide internal exception messages from users?
- Should the Web UI be split into static template/assets if it grows?
