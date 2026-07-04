## Why

Bot integration currently has a structured query API for card retrieval, but it has no direct way to translate arbitrary user-provided text. A dedicated translation API lets bots reuse the same local service and DeepSeek credentials without overloading the RAG query endpoint.

## What Changes

- Add an arbitrary-text translation API that defaults to translating into Chinese.
- Support bidirectional or multi-direction translation when the caller explicitly provides source and target language intent.
- Return both plain translated text and bot-ready structured message blocks.
- Reuse existing DeepSeek runtime configuration and credential handling.
- Keep the browser Web UI unchanged; translation is exposed through backend/API surfaces only.
- Optionally add a CLI `translate` command for local smoke testing and operational debugging.

## Capabilities

### New Capabilities

- `translation-api`: Defines arbitrary-text translation requests, language defaults, DeepSeek-backed translation behavior, and structured bot-oriented responses.

### Modified Capabilities

- `web-ui`: Extend the local ASGI JSON API with a backend-only translation route while leaving the browser UI unchanged.
- `cli-interface`: Add an optional `translate` command for invoking the translation service outside the browser.

## Impact

- Affected modules likely include `rag_agent/llm.py`, a new translation service module, `rag_agent/web.py`, `rag_agent/__main__.py`, and tests for API/CLI behavior.
- No database, Chroma, embedding, retrieval, reranker, or card indexing behavior changes are required.
- No new provider dependency is expected if the implementation reuses the existing LangChain/OpenAI-compatible DeepSeek integration.
