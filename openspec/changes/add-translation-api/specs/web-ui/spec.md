## ADDED Requirements

### Requirement: Web app exposes JSON translation API

The system SHALL expose `POST /api/translate` for backend translation execution without adding translation controls to the browser index page.

#### Scenario: Valid translation API request
- **GIVEN** a JSON object containing non-empty `text`
- **WHEN** `POST /api/translate` is called
- **THEN** the system parses request options into a translation request
- **AND** executes the configured translation handler
- **AND** returns JSON containing `translation`, `source_lang`, `target_lang`, `warnings`, and `structured`

#### Scenario: Translation API does not alter index page
- **WHEN** a user sends `GET /`
- **THEN** the response remains the card RAG query page
- **AND** the page does not need translation-specific controls

### Requirement: Web translation API validates request parameters

The system SHALL validate translation text and structured block length options.

#### Scenario: Empty translation text
- **GIVEN** a request body with empty or whitespace-only `text`
- **WHEN** `POST /api/translate` is called
- **THEN** the response status is `400`
- **AND** the response includes an error message that text is required

#### Scenario: Translation block length out of range
- **GIVEN** `structured_max_block_chars` is outside the supported range
- **WHEN** `POST /api/translate` is called
- **THEN** the response status is `400`
- **AND** the response includes a validation error message

### Requirement: Web translation API keeps credentials outside requests

The system SHALL not accept DeepSeek API credentials in translation request bodies.

#### Scenario: Translation request provides text only
- **GIVEN** a translation request enables DeepSeek-backed translation
- **WHEN** the request is executed
- **THEN** the server reads DeepSeek credentials from runtime environment settings
- **AND** request-provided API key fields are ignored or rejected
