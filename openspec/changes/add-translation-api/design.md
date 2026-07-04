## Context

The project already exposes a local ASGI service with `/api/query` for RAG card retrieval and bot-oriented structured response blocks. DeepSeek integration is centralized through environment-based runtime settings and `create_deepseek_chat_model()`.

Translation is a separate user intent from retrieval. It should not load `cards.cdb`, Chroma, embeddings, or rerankers, and it should not require the browser Web UI to grow another workflow. Bot callers need a small HTTP API that accepts arbitrary text and returns a translated string plus bounded message blocks.

## Goals / Non-Goals

**Goals:**

- Translate arbitrary text through the existing DeepSeek-compatible chat model configuration.
- Default to Chinese translation when the caller does not specify a target language.
- Support mutual translation when the caller explicitly requests source and target languages.
- Return a stable JSON response suitable for direct bot delivery.
- Keep translation independent from card retrieval/indexing runtime dependencies.

**Non-Goals:**

- Do not modify the browser Web UI for translation.
- Do not implement a QQ bot or platform-specific SDK integration.
- Do not add a separate translation provider in the first version.
- Do not perform glossary-backed official card-name matching in the first version.
- Do not persist translation history.

## Decisions

### Decision: Add a dedicated translation service module

Create a translation-focused module, such as `rag_agent.translation_service`, with explicit request/response dataclasses and pure response formatting helpers.

Rationale: Translation has different inputs, validation, and prompt constraints from RAG retrieval. Keeping it separate prevents `/api/query` and `query_service.py` from becoming a mixed-purpose orchestration layer.

Alternative considered: Add translation flags to `/api/query`. This would overload a card retrieval endpoint with non-retrieval behavior and make bot callers reason about unrelated fields.

### Decision: Expose translation through `POST /api/translate`

The ASGI app should parse translation JSON separately from query JSON and route it to the translation service. The browser index page should not add translation controls.

Rationale: Bot integrations can call one clear endpoint while the existing local UI remains focused on card-effect search.

Alternative considered: Add a second browser form. The user explicitly said Web UI does not need translation, so this is unnecessary scope.

### Decision: Reuse DeepSeek runtime configuration

The translation service should call `create_deepseek_chat_model(Settings.from_env())` and therefore use `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, and `DEEPSEEK_TIMEOUT_SECONDS`.

Rationale: The project already has provider configuration and dependency isolation for DeepSeek-compatible calls. Reusing it avoids another credential path.

Alternative considered: Add `TRANSLATION_*` environment variables immediately. This is premature unless translation needs a different model, provider, or timeout after real usage.

### Decision: Use explicit prompt constraints instead of retrieval context

The translation prompt should instruct the model to translate only the supplied text, preserve formatting where practical, avoid adding commentary, and default the target language to Chinese when omitted.

Rationale: Translation should be predictable for bot delivery and should not invent card rulings, explanations, or extra context.

Alternative considered: Use the existing answer-generation prompt. That prompt is intentionally RAG-specific and includes candidate-card constraints that do not apply to arbitrary text.

### Decision: Return bot-ready structured blocks

The response should include a top-level `translation`, `source_lang`, `target_lang`, `warnings`, and a `structured` object with `blocks[*].text`. Long translated text may be split or truncated according to a caller-provided maximum block length.

Rationale: This mirrors the existing bot-oriented query response pattern without requiring bot code to parse prose.

Alternative considered: Return only a translated string. That is enough for simple clients but does not solve bounded bot message delivery.

## Risks / Trade-offs

- [Risk] LLM language detection may be wrong when `source_lang` is `auto` or omitted. -> Mitigation: include returned language metadata as best effort and allow callers to specify source/target language explicitly.
- [Risk] Translation of domain-specific card text may use inconsistent terminology. -> Mitigation: use a translation prompt that asks for game-text terminology consistency; defer glossary-backed translation to a later change.
- [Risk] Very long inputs can exceed model context or bot message limits. -> Mitigation: validate non-empty text, allow max block length control, and return clear errors for request sizes that are too large if implementation adds a hard limit.
- [Risk] DeepSeek latency affects bot responsiveness. -> Mitigation: reuse the configured timeout and keep translation independent from heavy local model loading.
