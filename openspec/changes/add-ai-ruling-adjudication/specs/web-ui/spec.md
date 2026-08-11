## ADDED Requirements

### Requirement: Web app exposes adjudication API

The system SHALL expose a JSON API endpoint for AI ruling adjudication without changing the existing `/api/query` retrieval endpoint.

#### Scenario: Valid adjudication request

- **GIVEN** a JSON request containing a non-empty scenario
- **WHEN** the adjudication API is called
- **THEN** the system executes the adjudication workflow
- **AND** returns JSON containing the original scenario, resolved cards, ygocdb FAQ and supplement evidence when available, OCG rules evidence when available, reasoning, answer, warnings, and simulation status

#### Scenario: Empty adjudication scenario

- **GIVEN** a JSON request containing an empty or whitespace-only scenario
- **WHEN** the adjudication API is called
- **THEN** the response status is `400`
- **AND** the response includes an error message that scenario text is required

### Requirement: Web UI provides adjudication mode

The system SHALL provide a browser-accessible adjudication mode separate from similar-card retrieval.

#### Scenario: User opens local Web UI

- **WHEN** the local Web UI page is loaded
- **THEN** the page exposes a way to enter an adjudication scenario
- **AND** existing similar-card retrieval controls remain available

#### Scenario: User submits adjudication scenario

- **GIVEN** the user entered a ruling scenario in the adjudication UI
- **WHEN** the scenario is submitted
- **THEN** the browser sends an adjudication API request
- **AND** renders the adjudication answer and evidence summary from the response

### Requirement: Web adjudication can request simulation

The system SHALL allow Web API callers to request simulation verification separately from document-grounded adjudication.

#### Scenario: Simulation not requested

- **GIVEN** an adjudication API request does not request simulation
- **WHEN** the request is processed
- **THEN** the response indicates simulation was not requested

#### Scenario: Simulation requested but unavailable

- **GIVEN** an adjudication API request requests simulation
- **AND** the simulator adapter is unavailable
- **WHEN** the request is processed
- **THEN** the response includes a simulator warning
- **AND** the document-grounded adjudication answer is still returned
