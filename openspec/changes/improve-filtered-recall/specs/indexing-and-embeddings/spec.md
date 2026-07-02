## ADDED Requirements

### Requirement: Chroma documents include filterable structured metadata

The system SHALL persist scalar and boolean decoded card metadata in Chroma documents so dense retrieval can apply structured metadata filters.

#### Scenario: Index rank four XYZ monster

- **GIVEN** a rank 4 XYZ monster card is converted to a retrieval document
- **WHEN** the document metadata is cleaned and stored in Chroma
- **THEN** metadata includes `card_kind` as `monster`
- **AND** includes `is_xyz` as `true`
- **AND** includes `rank` as `4`

#### Scenario: Index non-XYZ level four monster

- **GIVEN** a level 4 non-XYZ monster card is converted to a retrieval document
- **WHEN** the document metadata is cleaned and stored in Chroma
- **THEN** metadata includes `is_xyz` as `false`
- **AND** includes `decoded_level` as `4`
- **AND** does not expose rank `4`

### Requirement: Chroma index rebuild is required for new dense filters

The system SHALL document that existing Chroma indexes must be rebuilt to use newly added dense metadata filters.

#### Scenario: User upgrades from old index

- **GIVEN** an existing Chroma index was built before filterable structured metadata existed
- **WHEN** the user wants dense metadata filtering
- **THEN** documentation instructs the user to rebuild the index with `build-index --reset`
