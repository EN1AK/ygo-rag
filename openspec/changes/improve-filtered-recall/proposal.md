## Why

Structured query filtering now correctly narrows broad queries such as `效果是除外对手墓地卡的四星超量怪兽` to rank 4 XYZ monsters, but the downstream retrieval stage still truncates sparse and dense candidates too aggressively. Relevant cards such as `No.80 狂装霸王 狂想战曲王` can match the structure filter and still be missed because sparse returns only the final `top_k` and Chroma dense search runs over the full collection before post-filtering.

This change improves filtered recall by widening the candidate pool and making dense retrieval apply metadata filters inside Chroma.

## What Changes

- When structured filters are applied, retrieval SHALL request at least 100 candidates from sparse retrieval before hybrid fusion.
- Dense Chroma retrieval SHALL support metadata filters derived from structured query filters, so semantic search runs within matching candidates where the index metadata supports it.
- Chroma documents SHALL persist filterable decoded metadata required by structured filters, including card kind, monster subtype booleans, rank, level, link rating, and attribute.
- Existing CLI/API flags remain unchanged.
- Existing unfiltered retrieval behavior remains unchanged except for metadata additions in newly built indexes.
- If a Chroma index lacks the required metadata, dense retrieval SHALL fall back to the existing safe allow-list or post-filter behavior and report a warning where available.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `retrieval-ranking`: Structured-filtered retrieval uses a wider recall pool and dense metadata-filtered search before hybrid/rerank.
- `indexing-and-embeddings`: Chroma indexing persists filterable decoded card metadata needed by dense metadata filters.

## Impact

- Affected code:
  - `rag_agent/agent.py`
  - `rag_agent/retrieval/vector.py`
  - `rag_agent/cards.py`
  - `rag_agent/card_metadata.py`
  - `rag_agent/query_service.py`
- Tests:
  - add retrieval tests proving a candidate ranked below top 10 but inside top 100 can be returned/reranked
  - add vector-store tests proving metadata filters are passed to Chroma
  - add metadata tests for filterable boolean fields
- Runtime/data:
  - existing Chroma indexes may need rebuild to contain new metadata fields
  - no new external services or model dependencies
