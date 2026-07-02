## ADDED Requirements

### Requirement: System decodes CDB card structure metadata

The system SHALL decode YGOPro `cards.cdb` raw card fields into normalized structure metadata usable by retrieval filters.

#### Scenario: Decode monster kind and subtype

- **GIVEN** a card row whose raw type bitmask represents an XYZ monster
- **WHEN** card metadata is decoded
- **THEN** the decoded metadata identifies the card as `monster`
- **AND** includes `xyz` in its monster types

#### Scenario: Decode attribute

- **GIVEN** a card row whose raw attribute represents DARK
- **WHEN** card metadata is decoded
- **THEN** the decoded metadata exposes normalized attribute `dark`

### Requirement: System derives level rank and link values

The system SHALL derive numeric level, rank, or link values from raw CDB data according to the decoded monster type.

#### Scenario: XYZ monster uses rank

- **GIVEN** a decoded monster includes `xyz`
- **AND** its raw level value is `4`
- **WHEN** structure metadata is produced
- **THEN** the decoded metadata exposes `rank` as `4`

#### Scenario: Non-XYZ non-Link monster uses level

- **GIVEN** a decoded monster does not include `xyz` or `link`
- **AND** its raw level value is `4`
- **WHEN** structure metadata is produced
- **THEN** the decoded metadata exposes `level` as `4`

#### Scenario: Link monster uses link rating

- **GIVEN** a decoded monster includes `link`
- **AND** its raw level-derived value represents link rating `2`
- **WHEN** structure metadata is produced
- **THEN** the decoded metadata exposes `link_rating` as `2`

### Requirement: System parses Chinese structure constraints

The system SHALL parse common Chinese and numeric card-structure constraints from the user query into normalized filters.

#### Scenario: Parse rank four XYZ monster

- **GIVEN** the query `效果是除外或者回收对手墓地的卡的四星超量怪兽`
- **WHEN** structured query parsing runs
- **THEN** the parsed filters include `card_kind` `monster`
- **AND** include monster type `xyz`
- **AND** include `rank` `4`

#### Scenario: Parse dark attribute monster

- **GIVEN** the query `有没有效果类似暗黑武装龙且是暗属性的卡`
- **WHEN** structured query parsing runs
- **THEN** the parsed filters include attribute `dark`

#### Scenario: No structure phrase

- **GIVEN** a query that contains only effect text and no recognized structure phrase
- **WHEN** structured query parsing runs
- **THEN** no structured filters are produced

### Requirement: Parser separates effect query from filters

The system SHALL provide an effect query suitable for lexical and vector retrieval after recognized structural phrases are parsed.

#### Scenario: Remove parsed structure terms

- **GIVEN** the query `效果是除外或者回收对手墓地的卡的四星超量怪兽`
- **WHEN** structured query parsing runs
- **THEN** the effect query preserves the effect intent `除外或者回收对手墓地的卡`
- **AND** the effect query does not rely on `四星超量怪兽` for ranking

### Requirement: System exposes filter diagnostics

The system SHALL expose structured filter diagnostics for query execution.

#### Scenario: Filters are parsed

- **GIVEN** a query contains recognized structure constraints
- **WHEN** query execution completes
- **THEN** the response includes the parsed filters
- **AND** includes the normalized effect query
- **AND** includes candidate counts before and after filtering when available

#### Scenario: No filters are parsed

- **GIVEN** a query contains no recognized structure constraints
- **WHEN** query execution completes
- **THEN** the response indicates that no structured filters were applied

## Open Questions

- Should a zero-match filter return no results or fall back to unfiltered retrieval with a warning?
- Should `四星超量怪兽` always be accepted as rank 4 XYZ wording?
- Which monster race aliases are required in the first version?
