## ADDED Requirements

### Requirement: Adjudication agent budgets are configurable by category
The system SHALL allow adjudication agent loop limits to distinguish total planner turns, retrieval calls, critique calls, and submit attempts.

#### Scenario: Default categorized budgets are loaded
- **GIVEN** no adjudication budget environment variables are configured
- **WHEN** settings are loaded
- **THEN** the agent SHALL have default limits for total turns, retrieval calls, critique calls, and submit attempts
- **AND** those limits SHALL allow at least one final submit attempt after ordinary retrieve-and-critique cycles

#### Scenario: Budget environment variables are configured
- **GIVEN** environment variables configure adjudication retrieval, critique, submit, or total limits
- **WHEN** settings are loaded
- **THEN** the configured values SHALL be reflected in runtime settings
- **AND** invalid non-positive values SHALL be rejected or normalized consistently with existing settings validation
