## ADDED Requirements

### Requirement: Adjudication runs through an iterative agent loop
The system SHALL support an adjudication control flow where the LLM planner chooses one next tool action per iteration from the current work state.

#### Scenario: Planner selects a tool
- **GIVEN** an adjudication scenario and an initialized agent work state
- **WHEN** the agent loop requests the next action
- **THEN** the planner SHALL return a structured action containing an allowed tool name, JSON arguments, and a short rationale
- **AND** the system SHALL execute only the selected allowed tool

#### Scenario: Tool observation updates state
- **GIVEN** the planner selected an allowed tool action
- **WHEN** the tool finishes
- **THEN** the system SHALL append the tool call and observation to the tool history
- **AND** the updated state SHALL be provided to the next planner iteration

### Requirement: Agent-generated retrieval queries are mechanism specific
The system SHALL allow the planner to generate distinct query strings for each FAQ or rules retrieval tool call.

#### Scenario: Planner searches rules for a mechanism
- **GIVEN** the current work state indicates an unresolved mechanism question
- **WHEN** the planner selects `search_rules`
- **THEN** the tool arguments SHALL include the planner-generated query string
- **AND** the rules retrieval SHALL use that query string rather than a single global mechanism query

#### Scenario: Planner refines a previous query
- **GIVEN** a prior retrieval observation returned insufficient evidence
- **WHEN** the planner selects another retrieval tool
- **THEN** the planner MAY provide a different query targeted at the remaining evidence gap
- **AND** the trace SHALL preserve both the old and new query strings

### Requirement: Agent work state distinguishes facts from hypotheses
The system SHALL separate verified tool results from planner notes, hypotheses, and intended next steps.

#### Scenario: Planner proposes a card mention
- **GIVEN** the planner believes a string may be a card name
- **WHEN** the planner calls a card-related tool
- **THEN** the string SHALL be treated as a query or mention candidate
- **AND** it MUST NOT become a resolved card unless the card-resolution tool validates it

#### Scenario: Planner records a hypothesis
- **GIVEN** the planner explains a possible ruling mechanism
- **WHEN** the explanation is stored in the agent state
- **THEN** the explanation SHALL be marked as planner notes or hypothesis
- **AND** it MUST NOT be treated as cited evidence

### Requirement: Loop termination is explicit and bounded
The system SHALL stop the agent loop only through accepted final submission, configured step limit, unrecoverable planner failure, or fallback routing.

#### Scenario: Planner submits final ruling
- **GIVEN** the planner has gathered evidence
- **WHEN** it selects `submit_ruling`
- **THEN** the system SHALL validate the submitted answer
- **AND** the loop SHALL terminate only if validation succeeds

#### Scenario: Step limit reached
- **GIVEN** the loop has reached the configured maximum number of iterations
- **WHEN** no valid final ruling has been submitted
- **THEN** the loop SHALL stop
- **AND** the system SHALL return a fallback answer over the gathered evidence with a warning

### Requirement: Agent trace exposes planner and tool steps
The system SHALL expose an ordered trace of planner decisions, tool calls, observations, validation errors, and final submission status.

#### Scenario: Trace is requested
- **WHEN** an adjudication response includes agent trace output
- **THEN** each loop iteration SHALL include the planner action, sanitized tool arguments, observation summary, evidence ids added, unresolved card state, and warnings
