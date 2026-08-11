## ADDED Requirements

### Requirement: Adjudication parses concrete scenarios

The system SHALL parse a user-provided ruling scenario into a structured representation before evidence retrieval.

#### Scenario: Scenario contains multiple card mentions and questions

- **GIVEN** a Chinese scenario mentions multiple card names, effect activations, a destruction object, a replacement candidate, and multiple asked questions
- **WHEN** adjudication starts
- **THEN** the system extracts card mentions, role labels, effect actions, relevant objects, replacement-effect candidates, and individual question items
- **AND** the original scenario text remains available in the adjudication response

#### Scenario: Scenario parsing is ambiguous

- **GIVEN** the scenario contains an ambiguous card mention or unclear controller role
- **WHEN** adjudication starts
- **THEN** the system includes a warning describing the ambiguity
- **AND** the system does not silently discard the ambiguous item

### Requirement: Adjudication resolves mentioned cards

The system SHALL resolve parsed card mentions against the local card database before ruling reasoning.

#### Scenario: Mention resolves to one card

- **GIVEN** the local card database contains exactly one matching card for a parsed mention
- **WHEN** card resolution runs
- **THEN** the resolved card includes card id, card name, and original effect text

#### Scenario: Mention has multiple plausible matches

- **GIVEN** the local card database contains multiple plausible matches for a parsed mention
- **WHEN** card resolution runs
- **THEN** the adjudication response includes all plausible matches
- **AND** the final answer marks the issue as needing disambiguation unless later context resolves the ambiguity

### Requirement: Adjudication retrieves exact card effects

The system SHALL include exact effect text for resolved involved cards as evidence.

#### Scenario: Resolved cards are available

- **GIVEN** a scenario resolves to one or more local cards
- **WHEN** evidence gathering runs
- **THEN** each resolved card contributes a card-effect evidence item with a stable evidence id
- **AND** the evidence item preserves original card effect text

### Requirement: Adjudication retrieves official rulings

The system SHALL retrieve ruling entries for involved cards from a local ygocdb ruling index when that index is configured.

#### Scenario: Ruling exists for involved card

- **GIVEN** the ruling index contains a ygocdb FAQ entry whose primary card matches an involved card
- **WHEN** evidence gathering runs
- **THEN** the adjudication response includes that ruling as official ruling evidence
- **AND** the evidence item identifies the matched card relationship as a direct-card match

#### Scenario: Ruling index is unavailable

- **GIVEN** no ruling index is configured or loaded
- **WHEN** evidence gathering runs
- **THEN** the adjudication response includes a warning that official ruling evidence was unavailable
- **AND** adjudication may continue with card effects and rule-document evidence

### Requirement: Adjudication normalizes ygocdb FAQ evidence

The system SHALL normalize ygocdb `faqs` entries into stable ruling evidence documents.

#### Scenario: FAQ entry is ingested

- **GIVEN** a ygocdb API response contains a `faqs` entry with `fid`, `title`, `date`, `question`, `answer`, and `refer`
- **WHEN** ruling ingestion runs
- **THEN** the system creates an evidence document with id `ygocdb:faq:{fid}`
- **AND** stores the API response card id as the primary card id
- **AND** stores the `refer` card ids as referenced card ids
- **AND** stores both HTML source fields and plain-text fields for retrieval

#### Scenario: FAQ appears from multiple card responses

- **GIVEN** the same `fid` appears while ingesting multiple ygocdb card responses
- **WHEN** ruling ingestion runs
- **THEN** the system deduplicates the FAQ by `fid`
- **AND** preserves all observed primary or referenced card relationships needed for lookup

### Requirement: Adjudication indexes ygocdb refer relationships

The system SHALL use ygocdb `faqs[*].refer` as a reverse index for related-card ruling lookup.

#### Scenario: Involved card appears in FAQ refer

- **GIVEN** a FAQ is returned from one card's ygocdb API response
- **AND** the FAQ `refer` object contains an involved card id
- **WHEN** evidence gathering runs for that involved card
- **THEN** the FAQ is returned as related-card ruling evidence
- **AND** the final answer does not label it as a direct-card ruling

#### Scenario: HTML links contain card references

- **GIVEN** a FAQ question or answer contains HTML links with `data-refer` card ids
- **WHEN** ruling ingestion runs
- **THEN** those links may be stored as secondary reference metadata
- **AND** the API `refer` object remains the authoritative related-card list when present

### Requirement: Adjudication normalizes ygocdb supplement evidence

The system SHALL normalize ygocdb `supplement` entries as lower-priority card supplement evidence.

#### Scenario: Supplement entry is available

- **GIVEN** a ygocdb API response contains `supplement.text`, `supplement.date`, and `supplement.refer`
- **WHEN** evidence gathering runs for that card
- **THEN** the system may include a `card_supplement` evidence item
- **AND** the evidence item stores both HTML source text and plain text
- **AND** the final answer does not treat supplement evidence as a concrete FAQ ruling

### Requirement: Adjudication retrieves analogous rulings

The system SHALL retrieve analogous ruling entries using related cards, similar effects, and mechanism-level query terms.

#### Scenario: Ruling lists related cards

- **GIVEN** a ruling entry is attached to one card but lists another involved card as a related or similar card
- **WHEN** analogous ruling retrieval runs
- **THEN** the ruling is returned as related-card evidence
- **AND** it is labeled separately from an exact-card ruling

#### Scenario: Similar mechanism is found

- **GIVEN** no exact ruling exists for an involved card
- **AND** the ruling index contains rulings for cards with similar effect mechanisms
- **WHEN** analogous ruling retrieval runs
- **THEN** the response includes those rulings as analogous evidence
- **AND** the final answer does not describe them as direct official rulings for the involved card

### Requirement: Adjudication retrieves OCG rule-document passages

The system SHALL retrieve relevant passages from the local OCG rules EPUB documentation during adjudication.

#### Scenario: Rules document index is available

- **GIVEN** the local OCG rules document index is configured
- **WHEN** evidence gathering runs for a scenario involving replacement, destruction, or effect resolution
- **THEN** the system retrieves relevant rule passages with stable evidence ids and section references

#### Scenario: Rules document index is unavailable

- **GIVEN** the local OCG rules document index is unavailable
- **WHEN** evidence gathering runs
- **THEN** the adjudication response includes a warning that rule-document evidence was unavailable

### Requirement: Adjudication parses OCG rules EPUB sections

The system SHALL parse the local `ocg-rule-readthedocs-io-zh-cn-latest.epub` into section-level rule evidence using EPUB table-of-contents metadata.

#### Scenario: EPUB table of contents is available

- **GIVEN** the rules EPUB contains `nav.xhtml` or `toc.ncx`
- **WHEN** rules ingestion runs
- **THEN** the system preserves chapter hierarchy, XHTML path, section anchor, and section title
- **AND** creates stable rule evidence ids from the path and anchor

#### Scenario: Rule section is very long

- **GIVEN** a TOC section contains text too long for one retrieval document
- **WHEN** rules ingestion runs
- **THEN** the system may split the section into secondary chunks
- **AND** each secondary chunk retains the original section path and anchor metadata

### Requirement: Ruling and rule retrieval are complementary

The system SHALL attempt official ruling retrieval and OCG rule-document retrieval for the same adjudication request instead of making rule retrieval only a fallback after ruling retrieval fails.

#### Scenario: Exact ruling exists

- **GIVEN** an exact official ruling exists for an involved card
- **WHEN** adjudication evidence gathering runs
- **THEN** the system still attempts to retrieve relevant OCG rule-document passages

#### Scenario: No exact ruling exists

- **GIVEN** no exact official ruling exists for an involved card
- **WHEN** adjudication evidence gathering runs
- **THEN** the system still attempts analogous ruling retrieval and OCG rule-document retrieval

### Requirement: Adjudication output is evidence-grounded

The system SHALL produce a structured adjudication answer whose legal claims cite gathered evidence items.

#### Scenario: Evidence supports a direct answer

- **GIVEN** gathered evidence contains card effects, ruling entries, or rule passages relevant to each asked question
- **WHEN** final adjudication is generated
- **THEN** the answer includes direct conclusions for each asked question
- **AND** each reasoning step cites one or more evidence ids
- **AND** the answer includes a confidence value

#### Scenario: Evidence is insufficient

- **GIVEN** gathered evidence is missing required card effects, rulings, or rule passages for a question
- **WHEN** final adjudication is generated
- **THEN** the answer marks the unsupported portion as uncertain
- **AND** the response lists the missing evidence or clarification needed

### Requirement: Adjudication distinguishes evidence kinds

The system SHALL label evidence by kind so users can distinguish direct rulings, analogous rulings, rule passages, card text, and simulation observations.

#### Scenario: Mixed evidence is gathered

- **GIVEN** evidence gathering returns card text, exact rulings, analogous rulings, and rule passages
- **WHEN** the adjudication response is built
- **THEN** each evidence item includes a kind field
- **AND** the final answer labels analogous evidence as analogous rather than exact

### Requirement: Simulation verification is optional and deferred

The system SHALL support an adjudication response shape that can include simulator observations, while the initial implementation may report simulation as unavailable.

#### Scenario: Simulation is not enabled

- **GIVEN** adjudication is requested without simulation enabled
- **WHEN** the response is built
- **THEN** the simulation field indicates that simulation was not requested
- **AND** document-grounded adjudication still returns normally

#### Scenario: Simulation adapter cannot encode scenario

- **GIVEN** simulation is requested
- **AND** the scenario cannot be encoded into a local engine setup
- **WHEN** simulation verification runs
- **THEN** the adjudication response includes a simulator warning
- **AND** the final answer remains based on document evidence

### Requirement: Adjudication decodes card effect tags

The system SHALL decode card effect tags from `datas.category` using `strings.conf` `!system 1100+` labels when available.

#### Scenario: strings.conf labels are available

- **GIVEN** a local `strings.conf` file contains `!system 1100+` effect tag labels
- **WHEN** a card document or adjudication card evidence is built
- **THEN** the system decodes the card's category bitmask using those labels
- **AND** includes decoded effect tags in card metadata or evidence metadata

#### Scenario: Known category bitmask is decoded

- **GIVEN** `完美电子多元驱动蛇·神龙` has category bitmask `0x3400002`
- **WHEN** effect tags are decoded
- **THEN** the decoded labels include `怪兽破坏`, `LP伤害`, `破坏耐性`, and `效果耐性`

### Requirement: Adjudication uses effect tags for retrieval

The system SHALL use decoded card effect tags to improve card, FAQ, and rule-document retrieval.

#### Scenario: Resolved cards have effect tags

- **GIVEN** adjudication resolves one or more involved cards with category bitmasks
- **WHEN** mechanism-level FAQ or rules retrieval runs
- **THEN** the retrieval query includes decoded effect tag labels from those cards

#### Scenario: FAQ search has related card tag overlap

- **GIVEN** an FAQ is associated with a primary or related card whose decoded effect tags overlap the scenario's resolved cards
- **WHEN** analogous FAQ search ranks candidates
- **THEN** the matching tag overlap may boost that FAQ's retrieval score

### Requirement: Adjudication parses OCG rule anchors precisely

The system SHALL extract OCG rules EPUB passages at the anchor-local section level instead of using the entire XHTML page for every anchor.

#### Scenario: Multiple anchors share an XHTML file

- **GIVEN** a rules XHTML file contains multiple TOC anchors
- **WHEN** rules ingestion parses one anchor
- **THEN** the resulting rule passage contains only that anchor's local section text
- **AND** does not include text from the next peer section

#### Scenario: Rule section is long

- **GIVEN** an anchor-local rule section is longer than the configured chunk size
- **WHEN** rules ingestion chunks the section
- **THEN** chunks are grouped by paragraph or sentence boundaries where possible
- **AND** each chunk keeps the original title, chapter path, XHTML path, and anchor metadata

#### Scenario: XHTML file has both top entry and anchor entries

- **GIVEN** the rules TOC contains a page-level entry and concrete anchor entries for the same XHTML file
- **WHEN** rules ingestion builds section documents
- **THEN** the page-level duplicate is skipped

### Requirement: Adjudication final answer is structured JSON

The system SHALL require LLM adjudication answers to return validated JSON with fixed fields and known evidence ids.

#### Scenario: LLM answer is generated

- **GIVEN** final adjudication uses an LLM
- **WHEN** the final answer prompt is built
- **THEN** the prompt requires JSON fields `conclusion`, `answers`, `evidence_use`, and `uncertainties`
- **AND** requires each key judgment to cite evidence ids from gathered evidence

#### Scenario: User asks a hypothetical branch

- **GIVEN** a parsed question contains a hypothetical branch such as "如果"
- **WHEN** the final answer is generated
- **THEN** the corresponding answer item includes both the actual-case conclusion and the hypothetical-branch conclusion

#### Scenario: LLM cites unknown evidence

- **GIVEN** the final answer JSON cites an evidence id not present in gathered evidence
- **WHEN** answer validation runs
- **THEN** the system rejects that answer and falls back or retries instead of returning the invalid citation as authoritative

### Requirement: Ruling data downloads avoid system proxy by default

The system SHALL avoid using system proxy settings by default when downloading ygocdb ruling data or `strings.conf`.

#### Scenario: Ruling download runs with default options

- **WHEN** the user downloads ygocdb ruling API payloads without enabling proxy use
- **THEN** the downloader uses a no-proxy opener

#### Scenario: strings.conf is downloaded

- **WHEN** the project downloads `strings.conf` for effect tag labels
- **THEN** the download uses a no-proxy opener unless proxy usage is explicitly enabled
