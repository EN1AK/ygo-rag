# card-database Specification

## Purpose

Define the current behavior for reading and inspecting YGOPro `cards.cdb` SQLite databases.

## Requirements

### Requirement: Database file must exist before reading

The system SHALL raise a file-not-found error when a requested `cards.cdb` path does not exist.

#### Scenario: Missing database path

- **GIVEN** a `cards.cdb` path that does not exist
- **WHEN** the system attempts to inspect or load that database
- **THEN** it reports that `cards.cdb` was not found for that path

### Requirement: Database inspection validates required tables

The system SHALL inspect the `datas` and `texts` tables before loading card records.

#### Scenario: Required tables exist

- **GIVEN** a SQLite database containing `datas` and `texts`
- **WHEN** the database is inspected
- **THEN** the system reports row counts and column names for both tables

#### Scenario: Required table is missing

- **GIVEN** a SQLite database missing `datas` or `texts`
- **WHEN** the database is inspected
- **THEN** the system raises an error naming the missing expected table

### Requirement: Card records are loaded from texts joined with datas

The system SHALL load cards by selecting rows from `texts` left-joined with `datas` on card id.

#### Scenario: Load cards from valid database

- **GIVEN** a valid YGOPro-style database
- **WHEN** cards are loaded
- **THEN** each card includes id, name, description, type, race, attribute, attack, defense, level, and category fields when available
- **AND** cards are ordered by `texts.id`

### Requirement: Blank card names are ignored

The system SHALL skip records where `texts.name` is null or empty.

#### Scenario: Text row has no card name

- **GIVEN** a row in `texts` with a null or empty name
- **WHEN** cards are loaded
- **THEN** that row is not returned as a card

## Open Questions

- Should the loader validate required columns, not only required tables?
- Should raw integer metadata such as `type`, `race`, `attribute`, and `category` be decoded into readable names?
- Should alias/setcode/ot fields be included in the domain model?
