## Why

The adjudication agent currently follows a mostly fixed workflow where code decides when to extract cards, retrieve evidence, check coverage, and retry. This limits ruling quality because the LLM cannot choose mechanism-specific searches, targeted card recovery, or alternative rules queries based on observations from prior tool calls.

## What Changes

- Replace the fixed `extract -> resolve_cards -> retrieve_evidence -> coverage -> final` adjudication control flow with an `agent -> tool -> agent` loop.
- Add a constrained LLM planner that selects the next allowed tool and tool arguments from the current work state.
- Expose card resolution, card search, direct FAQ search, related FAQ search, analogous FAQ search, rules search, coverage critique, and final ruling submission as auditable tools.
- Let the LLM generate mechanism-specific FAQ and rules queries per tool call instead of relying on one shared retrieval query.
- Preserve existing card-resolution validity gates and staged evidence relationship semantics inside tool implementations.
- Add a `submit_ruling` tool that validates final JSON structure, evidence citations, unresolved card state, and confidence constraints before accepting the final answer.
- Keep the existing workflow as a fallback or compatibility path during migration.

## Capabilities

### New Capabilities

- `adjudication-agent-loop`: Defines the iterative LLM planner loop, work-state memory, tool-call protocol, stop conditions, and final submission behavior.
- `adjudication-tool-contracts`: Defines the adjudication tools available to the loop, including argument schemas, observations, evidence persistence, and trace requirements.

### Modified Capabilities

- `answer-generation`: Final adjudication answers are submitted through a validated tool call with strict JSON, evidence citations, and unresolved-card safeguards.

## Impact

- Affected code:
  - `rag_agent/adjudication_agent.py`
  - `rag_agent/adjudication.py`
  - `rag_agent/ruling_index.py`
  - CLI and web trace output that exposes agent steps, tool calls, observations, and final answer payloads
- Affected tests:
  - agent loop planning and tool dispatch tests
  - tool schema and observation tests
  - card recovery loop tests for unresolved and ambiguous mentions
  - mechanism-specific FAQ and rules query tests
  - final `submit_ruling` validation tests
  - compatibility tests for the existing workflow fallback
- No new external data source is required. Existing card database, FAQ index, rules index, LLM configuration, and validity gates remain reusable.
