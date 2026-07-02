## Why

Current retrieval treats structural card constraints such as `暗属性`, `四星`, `四阶`, and `超量怪兽` as plain text. This causes recall failures when the desired cards are best identified by CDB metadata first and effect text second, for example `效果是除外或者回收对手墓地的卡的四星超量怪兽`.

This change adds structured query filters so retrieval can narrow candidates by decoded card metadata before lexical, vector, and reranker scoring.

## What Changes

- Decode selected YGOPro `cards.cdb` `datas` fields into searchable card metadata:
  - card kind: monster, spell, trap
  - monster subtypes needed for filtering: normal, effect, fusion, ritual, synchro, xyz, pendulum, link
  - attribute and race where present
  - level/rank/link value derived from the existing `level` field according to monster type
- Parse common Chinese query constraints into a structured filter object:
  - attributes such as `暗属性`, `光属性`
  - monster kinds such as `超量`, `融合`, `同调`, `连接`, `仪式`
  - numeric constraints such as `四星`, `4星`, `四阶`, `4阶`
  - broad kind constraints such as `怪兽`, `魔法`, `陷阱`
- Apply parsed filters as a candidate prefilter before sparse, dense, hybrid, local rerank, and LLM rerank steps.
- Preserve existing retrieval behavior when no structured filters are parsed.
- Expose parsed filters and filter match diagnostics in CLI/API results for debugging and bot integration.
- Do not add new external services. CDB bitmask decoding may reference the public `EN1AK/yugioh-ccb` implementation, but the resulting behavior should be implemented locally in this project.

## Capabilities

### New Capabilities

- `structured-query-filters`: Parses card-structure constraints from user queries, decodes CDB card metadata, and filters retrieval candidates before ranking.

### Modified Capabilities

- `retrieval-ranking`: Retrieval must use structured candidate prefiltering before sparse/dense scoring when filters are parsed.
- `cli-interface`: CLI query output must expose parsed structured filters and filter warnings/diagnostics when present.
- `web-ui`: Web API responses must include structured filter data so other services can consume the same behavior as CLI.

## Impact

- Affected code:
  - `rag_agent/cards.py`
  - `rag_agent/db.py`
  - `rag_agent/agent.py`
  - `rag_agent/query_service.py`
  - `rag_agent/web.py`
  - `rag_agent/__main__.py`
  - retrieval modules under `rag_agent/retrieval/`
- Tests:
  - add unit tests for CDB metadata decoding
  - add unit tests for Chinese structured query parsing
  - update retrieval/query-service/API/CLI tests for prefilter behavior and exposed parsed filters
- Runtime behavior:
  - no new model dependency
  - no new database dependency
  - existing Chroma indexes may need rebuild or compatibility handling if decoded metadata is added to vector documents
