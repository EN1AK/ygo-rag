## Why

The current high-quality reranking path depends on a local `bge-reranker-v2-m3` model, which can be slow or impractical on low-compute machines and complicates server deployment. This change explores an optional LLM judge / LLM rerank path so deployments can trade local GPU/CPU load for DeepSeek API calls while keeping the existing local reranker available.

## What Changes

- Add an optional LLM-based reranking mode that scores or orders retrieved candidates using the configured DeepSeek-compatible LLM.
- Keep the existing local bge reranker unchanged and available.
- Allow query execution to choose between no rerank, local rerank, and LLM rerank.
- Return structured LLM judge evidence sufficient to audit why a candidate was promoted or demoted.
- Add CLI and Web UI controls for selecting LLM rerank when LLM credentials are configured.
- Add tests with fake LLM responses; do not require live DeepSeek calls in normal test runs.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `retrieval-ranking`: add optional LLM judge / LLM rerank as an alternative to the local reranker.
- `answer-generation`: define how LLM judging prompts and structured judge outputs behave separately from final answer synthesis.
- `runtime-configuration`: define environment/config behavior for selecting rerank provider and controlling LLM judge limits.
- `cli-interface`: expose LLM rerank selection in the query command.
- `web-ui`: expose LLM rerank selection in the local Web UI and API.

## Impact

- Affected modules likely include `rag_agent/agent.py`, `rag_agent/query_service.py`, `rag_agent/llm.py`, `rag_agent/__main__.py`, and `rag_agent/web.py`.
- Retrieval modules may need a new reranker implementation or protocol adapter for LLM judge scoring.
- Tests should cover parsing, provider selection, JSON response handling, CLI flags, and Web API parameters without making network calls.
- Runtime cost shifts from local reranker compute to LLM API latency and token usage when LLM rerank is enabled.
