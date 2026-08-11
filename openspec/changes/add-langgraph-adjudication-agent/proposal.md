## Why

The current adjudication workflow is a one-shot evidence pipeline: it parses a scenario, resolves cards mostly by direct matching, retrieves evidence once, and asks the LLM for a final answer. This is not yet a full agent tool because it cannot notice unresolved or fuzzy card mentions, call the existing card RAG path to resolve them, iterate on missing evidence, or verify that every question branch is covered before answering.

## What Changes

- Add a LangGraph-based adjudication agent loop for ruling scenarios.
- Use the existing card retrieval/RAG channel as a callable fuzzy card-resolution tool before final evidence gathering.
- Represent adjudication state explicitly, including unresolved mentions, resolved card candidates, selected card ids, evidence, question branches, coverage gaps, and tool-call trace.
- Add graph nodes for scenario extraction, fuzzy card search, card resolution, ruling FAQ retrieval, OCG rules retrieval, coverage checking, and final JSON answer synthesis.
- Iterate retrieval when required cards, direct/related rulings, analogous rulings, or hypothetical branches are not sufficiently covered.
- Keep evidence-grounded final answering: the LLM may synthesize the final ruling only from returned evidence ids, and the system must validate structured JSON output and cited evidence ids.
- Preserve a deterministic fallback path that reports evidence and gaps when the agent cannot resolve ambiguity or coverage.
- No breaking changes to existing query, translation, CLI, or Web behavior.

## Capabilities

### New Capabilities

- `adjudication-agent-loop`: Covers LangGraph state orchestration for ruling adjudication, fuzzy card-resolution tools using existing RAG retrieval, iterative evidence retrieval, coverage checks, tool trace, and structured final answers.

### Modified Capabilities

- None.

## Impact

- Affected modules likely include adjudication services under `rag_agent/`, retrieval helpers, prompt builders, CLI/Web adjudication wiring, and tests.
- Adds a LangGraph runtime dependency or optional dependency path for the adjudication agent.
- Reuses local `cards.cdb`, card effect tags, ygocdb cache, and parsed OCG rules EPUB data.
- May add response fields for agent state, coverage gaps, and tool-call trace while preserving existing adjudication response compatibility.
