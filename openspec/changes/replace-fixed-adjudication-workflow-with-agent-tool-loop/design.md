## Context

The current adjudication agent uses LangGraph, but most control decisions are still encoded as a fixed workflow: extract scenario, resolve cards, retrieve evidence, check coverage, optionally retry, then answer. This is predictable, but it prevents the LLM from choosing different retrieval strategies for different mechanisms. For example, a case about effect activation restrictions should trigger different rule queries from a case about destruction replacement or later effect processing.

Recent changes strengthened card-resolution validity and staged evidence relationships. Those safeguards should remain, but they should become tool internals used by an agent loop instead of fixed graph nodes.

## Goals / Non-Goals

**Goals:**

- Replace the fixed adjudication control path with an iterative `agent -> tool -> agent` loop.
- Let the LLM choose the next tool and generate mechanism-specific card, FAQ, and rules queries.
- Keep all factual acceptance in code: card resolution, evidence ids, relationships, and final answer validation remain constrained.
- Preserve traceability by recording every planner decision, tool call, tool observation, and final submission attempt.
- Reuse existing card database, persistent ruling indexes, rules index, card validity gates, and answer validators.
- Keep the existing fixed workflow available as a fallback during migration.

**Non-Goals:**

- Do not let the LLM invent card ids, evidence ids, aliases, or ruling facts.
- Do not implement ygoenv simulation in this change.
- Do not build a full game-state parser.
- Do not require new external data sources.
- Do not remove existing CLI/Web adjudication entry points.

## Decisions

### Decision: Use a small agent loop graph

The new graph will use a compact loop:

```text
agent_plan -> execute_tool -> agent_plan -> ... -> final
```

`agent_plan` asks the LLM for one structured action. `execute_tool` validates and executes that action. The loop ends when the action is `submit_ruling`, when a hard step limit is reached, or when a validation failure requires deterministic fallback.

Rationale: this keeps LangGraph useful for state transitions while moving query and tool sequencing decisions to the LLM.

Alternative considered: add more conditional edges to the current workflow. Rejected because it still encodes mechanism-specific retrieval order in code and would continue to centralize query generation in `_mechanism_query()`.

### Decision: LLM plans tool calls, code owns facts

The planner output will be strict JSON with an action enum and arguments. The LLM can propose mentions, queries, and rationale, but tool outputs are the only accepted source for resolved cards, evidence ids, and final answer citations.

Rationale: this matches the desired agent behavior without regressing into trusting parser output as game state.

### Decision: Expose retrieval as mechanism-specific tools

The tool layer will expose separate operations:

- `resolve_cards`
- `search_card`
- `search_direct_faq`
- `search_related_faq`
- `search_analogous_faq`
- `search_rules`
- `critique_coverage`
- `submit_ruling`

FAQ and rules tools accept LLM-generated query text. Direct and related FAQ tools may also filter by resolved card ids. Tool observations return concise structured results and add durable evidence to the state.

Rationale: the LLM can ask different questions as it learns from observations, while retrieval execution still uses existing indexes and metadata.

### Decision: Maintain a work-state memory

The loop state will track:

- scenario text and questions
- planner notes and hypotheses
- resolved, ambiguous, and unresolved card mentions
- evidence pool grouped by relationship and stage
- tool history with arguments and observations
- coverage gaps and warnings
- submit attempts and validation errors
- step count

Rationale: the planner needs enough context to decide whether to search again, recover card names, inspect rules, or submit.

### Decision: `submit_ruling` is a tool with validation

The final answer is accepted only through `submit_ruling`. The submitted payload must pass the existing structured JSON and citation validators, must cite only evidence in the pool, and must surface unresolved/ambiguous card mentions as uncertainty or low confidence unless explicitly resolved.

Rationale: final-answer validation should be a first-class step, not an unconstrained string returned at the end.

### Decision: Keep deterministic fallback

If the LLM is absent, emits invalid actions repeatedly, exceeds the step limit, or cannot produce a valid submission, the system will use the existing deterministic fallback answer over the gathered evidence.

Rationale: this protects CLI/Web behavior and keeps tests deterministic.

## Risks / Trade-offs

- More LLM autonomy can increase latency and token usage -> Use a configurable max step count, compact observations, and duplicate-query suppression.
- The planner may loop on similar searches -> Record normalized tool signatures and return a warning or forced critique after repeated calls.
- The planner may choose poor queries -> Let tools return low/no-hit observations and rely on the next loop, while tests cover representative mechanisms.
- Tool observations may grow too large -> Store full evidence in state but pass summarized observations to the planner.
- The new loop may make behavior less deterministic -> Use strict action schemas, fixture LLMs in tests, and retain fallback workflow.

## Migration Plan

1. Add the tool schema, planner response parser, and work-state structures without removing existing workflow code.
2. Wrap current card resolution and evidence retrieval functions as tools.
3. Add the `agent_plan -> execute_tool` loop and route adjudication through it behind a configuration flag or internal mode.
4. Add tests for card recovery, mechanism-specific query generation, staged evidence accumulation, loop limits, invalid actions, and final submission validation.
5. Switch the default adjudication agent path after parity tests pass.
6. Keep the previous workflow reachable as fallback until the agent loop is stable in CLI and Web traces.

## Open Questions

- Should the agent loop be enabled by default immediately, or gated behind a runtime setting for one release?
- Should `critique_coverage` be a deterministic tool, an LLM self-critique prompt, or both?
- How much of tool history should be included in every planner prompt before summarization is required?
- Should repeated no-result queries be blocked by exact signature or by semantic similarity?
