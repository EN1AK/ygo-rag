## ADDED Requirements

### Requirement: Translation service accepts arbitrary text

The system SHALL accept arbitrary non-empty text for translation without requiring card database, retrieval, embedding, Chroma, or reranker resources.

#### Scenario: Translate plain text
- **GIVEN** a translation request with non-empty `text`
- **WHEN** the translation service executes
- **THEN** it calls the configured DeepSeek-compatible chat model
- **AND** it does not load `cards.cdb`, Chroma, embeddings, or rerankers

#### Scenario: Empty translation text
- **GIVEN** a translation request with empty or whitespace-only `text`
- **WHEN** the request is validated
- **THEN** the system rejects the request with a validation error stating that text is required

### Requirement: Translation defaults to Chinese

The system SHALL translate to Chinese by default when the caller does not specify a target language.

#### Scenario: Target language omitted
- **GIVEN** a translation request without `target_lang`
- **WHEN** the request is parsed
- **THEN** the target language defaults to `zh-CN`

#### Scenario: Source language omitted
- **GIVEN** a translation request without `source_lang`
- **WHEN** the request is parsed
- **THEN** the source language defaults to `auto`

### Requirement: Translation supports explicit language direction

The system SHALL support mutual translation when callers explicitly provide source and target language intent.

#### Scenario: Explicit English to Chinese
- **GIVEN** a translation request with `source_lang` set to `en`
- **AND** `target_lang` set to `zh-CN`
- **WHEN** translation executes
- **THEN** the prompt instructs the model to translate from English to Chinese

#### Scenario: Explicit Chinese to English
- **GIVEN** a translation request with `source_lang` set to `zh-CN`
- **AND** `target_lang` set to `en`
- **WHEN** translation executes
- **THEN** the prompt instructs the model to translate from Chinese to English

### Requirement: Translation prompt preserves user text meaning

The system SHALL instruct the model to translate only the supplied text while preserving meaning, formatting, and domain terminology where practical.

#### Scenario: Build translation prompt
- **GIVEN** translation text and language settings
- **WHEN** the translation prompt is built
- **THEN** it includes the source language, target language, and original text
- **AND** it instructs the model not to add commentary outside the translation

### Requirement: Translation response includes bot-ready structure

The system SHALL return translated text and structured message blocks suitable for bot delivery.

#### Scenario: Translation succeeds
- **GIVEN** the model returns translated text
- **WHEN** the translation response is serialized
- **THEN** the response includes `translation`
- **AND** `source_lang`
- **AND** `target_lang`
- **AND** `warnings`
- **AND** `structured.blocks`

#### Scenario: Bot sends structured block text
- **GIVEN** a translation response with structured blocks
- **WHEN** a bot reads `structured.blocks[*].text`
- **THEN** each block contains ready-to-send translated message text

### Requirement: Translation structured output supports block length control

The system SHALL allow callers to bound structured translation block length.

#### Scenario: Max block length supplied
- **GIVEN** a translation response whose translated text exceeds `structured_max_block_chars`
- **WHEN** structured blocks are built
- **THEN** block text is split or truncated so each block respects the requested limit
- **AND** the response indicates whether any block was truncated or split
