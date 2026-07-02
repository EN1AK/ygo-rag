## Why

QQ bot messages have per-message length limits, while the current `/api/query` response primarily exposes one long `answer` string plus raw result items. Bot integration needs a stable structured response that can be split by card without parsing prose.

## What Changes

- Add a structured bot-oriented response shape to the existing query API.
- Include per-card message blocks that are safe to send independently.
- Preserve the current `answer`, `results`, and `warnings` fields for backward compatibility.
- Allow callers to request structured output with optional per-card text length limits.
- Keep final LLM answer generation optional; structured card blocks must be available without `llm=true`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `web-ui`: extend `POST /api/query` response contract for bot-oriented structured output.
- `answer-generation`: define per-card structured message content derived from retrieved candidates.

## Impact

- Affected modules likely include `rag_agent/query_service.py`, `rag_agent/web.py`, and tests for API response shape.
- No database, embedding, Chroma, or reranker behavior changes are required.
- Existing callers using `answer`, `results`, and `warnings` should continue to work.
