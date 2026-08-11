## Why

The adjudication agent currently builds FAQ and OCG rules retrieval structures at runtime from `data/ygocdb/*.json` and the local EPUB. This makes real ruling queries slow, makes analogous FAQ search weaker than card RAG, and forces each agent run to depend on ad hoc sparse matching rather than a reusable indexed evidence store.

## What Changes

- Add persistent RAG indexes for ygocdb FAQ/supplement evidence and OCG rules passages.
- Build vector indexes from local cached sources:
  - ygocdb `faqs` and `supplement` payloads under `data/ygocdb`
  - parsed OCG rules sections/passages from the local EPUB
- Store enough metadata for direct card, related card, analogous FAQ, supplement, and rule provenance.
- Reuse the existing card RAG retrieval pattern: sparse recall, vector recall, hybrid merge, optional local rerank, and optional LLM judge rerank.
- Add CLI/service commands to build, inspect, and query ruling indexes without requiring live network access.
- Update the adjudication agent evidence tools to prefer persistent indexes and fall back to runtime cache parsing when indexes are missing.
- Keep final adjudication evidence-grounded: retrieved documents must preserve stable evidence ids and source metadata for citation validation.
- No breaking changes to existing card query, adjudication, translation, CLI, or Web behavior.

## Capabilities

### New Capabilities

- `persistent-ruling-rag-index`: Covers persistent vector/hybrid indexes for ygocdb FAQ, card supplements, and OCG rules passages; indexed evidence retrieval; reranking; and adjudication-agent integration.

### Modified Capabilities

- `adjudication-agent-loop`: Evidence tools should use the persistent ruling/rules indexes when available before falling back to runtime FAQ/rules parsing.
- `cli-interface`: CLI should expose build/inspect/query commands and adjudication flags/configuration for persistent ruling indexes.
- `runtime-configuration`: Settings should include paths, collection names, retrieval limits, and rerank controls for ruling FAQ and OCG rules indexes.

## Impact

- Affected modules likely include `rag_agent/adjudication.py`, `rag_agent/adjudication_agent.py`, `rag_agent/adjudication_service.py`, retrieval helpers, config, CLI, Web/API response handling, and tests.
- Reuses Chroma/vector retrieval dependencies already used by card RAG.
- Adds persistent index artifacts under configured data directories, separate from the raw ygocdb JSON cache and EPUB file.
- Improves first-query latency after indexing and improves analogous ruling recall quality.
- Requires index build/refresh workflows when ygocdb cache files or the rules EPUB change.
