## 1. Translation Service

- [x] 1.1 Add translation request and response dataclasses with defaults for `source_lang=auto` and `target_lang=zh-CN`.
- [x] 1.2 Implement translation prompt construction that preserves formatting, avoids commentary, and supports explicit language direction.
- [x] 1.3 Implement DeepSeek-backed translation execution using existing runtime settings and `create_deepseek_chat_model()`.
- [x] 1.4 Implement structured translation response blocks with optional `structured_max_block_chars` handling.

## 2. HTTP API

- [x] 2.1 Add `POST /api/translate` routing to the ASGI app without changing the browser index page.
- [x] 2.2 Add translation request parsing and validation for text, language fields, and structured block length.
- [x] 2.3 Serialize translation responses with `translation`, `source_lang`, `target_lang`, `warnings`, and `structured`.
- [x] 2.4 Ensure translation requests do not accept or require API credentials in the request body.

## 3. CLI

- [x] 3.1 Add `translate` to top-level CLI help.
- [x] 3.2 Implement `python -m rag_agent translate <text>` with default Chinese translation.
- [x] 3.3 Add CLI options for explicit `--source-lang`, `--target-lang`, and optional structured block length if useful for bot testing.
- [x] 3.4 Reuse existing CLI error handling for invalid translation input and DeepSeek configuration failures.

## 4. Tests

- [x] 4.1 Add unit tests for translation request defaults and validation.
- [x] 4.2 Add unit tests for translation prompt construction with default and explicit language directions.
- [x] 4.3 Add unit tests for structured translation block formatting and length control.
- [x] 4.4 Add Web API tests for valid `/api/translate`, empty text, invalid block length, and unchanged `GET /` behavior.
- [x] 4.5 Add CLI tests for help output, default translation invocation with a fake translator, explicit target language, and empty input failure.

## 5. Documentation and Verification

- [x] 5.1 Document `/api/translate` request and response examples for bot integration.
- [x] 5.2 Document `translate` CLI usage if the command is implemented.
- [x] 5.3 Run the test suite with plugin autoload disabled.
- [x] 5.4 Run OpenSpec validation for `add-translation-api`.
