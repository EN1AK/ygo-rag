## Why

The adjudication agent currently resolves card names mostly through deterministic mention slicing plus local sparse lookup, while LLM parsing is best-effort and can fail without materially improving ambiguous or misspelled card references. This produces noisy traces, unresolved partial mentions, and insufficient agent-loop behavior when user scenarios contain fuzzy, abbreviated, or multi-card card-name phrases.

## What Changes

- Add an agentic card resolution loop for adjudication scenarios:
  - LLM extracts possible card-name strings from the scenario as structured candidates.
  - Local card RAG retrieves top-k candidates for each extracted string.
  - Rule-based selection chooses high-confidence matches, with optional LLM selection constrained to the local candidate set.
  - Unresolved or ambiguous mentions trigger targeted card-search iterations before evidence retrieval proceeds.
- Keep deterministic parsing and lookup as fallback paths when LLM extraction or selection fails.
- Record card-resolution trace data showing extraction source, retrieved candidates, selected cards, ambiguity, retries, and warnings.
- Avoid letting the LLM introduce card ids or card names that were not present in local card candidates.

## Capabilities

### New Capabilities
- `agentic-card-resolution`: Defines the adjudication agent loop for extracting possible card names, retrieving local card candidates, selecting resolved cards, and retrying unresolved/ambiguous mentions.

### Modified Capabilities
- `retrieval-ranking`: Card candidate retrieval shall support use as a constrained candidate provider for agentic card resolution.
- `answer-generation`: Adjudication answer generation shall consume the resolved-card set and card-resolution trace produced by the agentic loop.

## Impact

- Affected code:
  - `rag_agent/adjudication_agent.py`
  - `rag_agent/adjudication.py`
  - `rag_agent/retrieval/*`
  - adjudication service/CLI/Web paths that construct or expose agent traces
- Affected tests:
  - card mention extraction and resolution tests
  - agent loop and coverage tests
  - LLM failure/fallback tests
- Runtime dependencies:
  - Uses existing DeepSeek-compatible LLM integration when enabled.
  - Uses existing local card database and card retrieval indexes; no new external card source is introduced.
