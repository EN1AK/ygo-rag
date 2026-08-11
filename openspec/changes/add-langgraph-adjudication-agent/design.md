## Context

The current adjudication implementation is a linear service. It can parse a scenario, resolve exact card names, gather card text, ygocdb FAQ evidence, and OCG rules evidence, then ask an LLM to produce a structured answer. Recent improvements added effect tags, better EPUB section parsing, and stricter final-answer JSON prompting, but the system still behaves like a single API call rather than an agent.

The main gap is control flow. If a user writes `破械冥官·篁` while the local database stores `破械冥官 篁`, the current parser can miss the card unless a later LLM parser happens to normalize it. If evidence for a hypothetical branch is absent, the final answer prompt can ask the LLM to answer it, but the system does not loop back to search for better analogous rulings. A ruling agent needs explicit state, tool calls, gap detection, and bounded iteration.

Baseline capabilities already implemented by the preceding adjudication change include: `strings.conf`-based card effect tag decoding, effect tags in card documents and adjudication evidence, ygocdb FAQ/supplement indexes, anchor-local OCG rules EPUB parsing, paragraph-aware rules chunking, weighted rule title/chapter search, no-proxy local data downloads by default, and fixed-field JSON final answer validation. The LangGraph change should reuse these capabilities rather than reimplement them.

## Goals / Non-Goals

**Goals:**

- Add a LangGraph adjudication graph that turns ruling adjudication into an auditable agent loop.
- Use the existing card retrieval/RAG channel as a fuzzy card-resolution tool.
- Resolve card mentions through exact, normalized, and retrieval-backed candidate search.
- Retrieve direct, related, analogous FAQ, supplement, and OCG rules evidence through graph tools.
- Add a coverage check that decides whether the graph can answer, must search again, or must report ambiguity/gaps.
- Preserve evidence-grounded final JSON answers and evidence-id validation.
- Provide a tool-call trace suitable for CLI/Web debugging and tests.

**Non-Goals:**

- Do not implement `ygoenv` simulation in this change.
- Do not replace the existing one-shot adjudication pipeline immediately; the graph can be introduced as an opt-in or internally selected path.
- Do not require live network access for normal adjudication; ygocdb and rules evidence should continue to work from local caches.
- Do not allow the LLM to invent card ids, card text, FAQ entries, or rule passages outside tool results.

## Decisions

### Decision: Use LangGraph as the orchestration layer

Create an adjudication graph with typed state rather than embedding a hand-written loop inside `RulingAdjudicator`. LangGraph fits the problem because each stage is stateful and conditional: unresolved mentions route to card search, low evidence coverage routes to targeted retrieval, and sufficient coverage routes to final answer synthesis.

Alternative considered: keep adding logic to the current linear service. That is simpler short-term but makes loop state, retries, trace, and branch coverage harder to test and reason about.

### Decision: Model graph state explicitly

Use a state object such as `AdjudicationAgentState` with:

- original scenario text
- parsed scenario fields and questions
- raw mention candidates
- unresolved mentions
- card candidate lists per mention
- selected resolved cards
- effect tags and mechanism terms
- evidence items
- coverage report
- tool trace
- warnings and errors
- loop counters
- final answer JSON

The state should be serializable enough for tests and Web/CLI diagnostics.

Alternative considered: pass mutable local variables between node functions. That makes traces and retries harder to inspect.

### Decision: Split existing adjudication behaviors into tools

The graph should call small deterministic tools:

- `extract_scenario`: structured scenario extraction, initially using existing parser/LLM parser.
- `search_cards`: fuzzy card search using exact match, normalized match, sparse card retriever, and optional dense retrieval.
- `select_card_candidates`: choose high-confidence matches or mark ambiguity.
- `get_card_effects`: create card effect evidence for resolved cards.
- `get_direct_related_faqs`: retrieve direct and refer-based ygocdb FAQ.
- `search_analogous_faqs`: mechanism and effect-tag based FAQ search.
- `search_ocg_rules`: rules EPUB search.
- `check_coverage`: determine missing cards, missing evidence, missing hypothetical branch evidence, and ambiguity.
- `final_answer`: evidence-constrained JSON answer generation.

These are graph tools, not Codex skills. They run inside the application and can be unit-tested.

Alternative considered: expose one generic retrieval tool to the LLM and let it decide queries. That is more flexible but less controllable and makes hallucinated tool planning harder to detect.

### Decision: Fuzzy card resolution uses existing RAG before final evidence retrieval

Card name resolution should first try exact/normalized matching, then call the existing card retrieval path over card name, effect text, type metadata, and effect tags. Normalization should remove or standardize spaces, middle dots, full-width punctuation, brackets, and common separators. Candidate selection should use confidence thresholds:

- one high-confidence normalized or retrieval result -> resolved
- multiple close candidates -> ambiguous
- no viable candidate -> unresolved

The graph should not silently pick a low-confidence candidate.

Alternative considered: prompt-only fuzzy matching. Prompt text cannot reliably resolve card ids unless the model can call a retrieval tool and inspect candidates.

### Decision: Coverage check controls iteration

After each retrieval pass, `check_coverage` decides:

- continue to card search if any likely card mention is unresolved
- continue to analogous FAQ/rules search if a question branch lacks evidence
- continue to mechanism search if final answer needs an interaction such as "target not destroyed, later special summon"
- finalize if every question has enough evidence or if retry limits are reached

Coverage is not a claim of correctness; it is a guard that prevents obviously incomplete answers.

Alternative considered: always run a fixed set of searches. That wastes work and still does not know when a hypothetical branch was missed.

### Decision: Keep final answering evidence-only and JSON-validated

The final node should reuse the structured JSON answer prompt and validation, but the prompt context should include the coverage report and tool trace summary. Every key judgment must cite known evidence ids. If the LLM returns invalid JSON or unknown citations, the graph should either retry once with a repair prompt or fall back to a deterministic evidence/gap summary.

Alternative considered: accept free-form LLM output. That is less useful for Web display, tests, and downstream validation.

### Decision: Bound loops and expose trace

The graph must use max iteration limits and deterministic routing. The response should include a compact trace such as:

- node name
- tool input summary
- candidate/evidence counts
- decision or coverage status
- warnings

This trace is important because ruling failures are often retrieval or ambiguity failures, not only final-answer failures.

Alternative considered: hide internal steps. That makes it hard to diagnose why a card or FAQ was missed.

## Risks / Trade-offs

- [Risk] LangGraph adds a new dependency and mental model -> Mitigation: isolate graph construction in a small module and keep existing service code reusable.
- [Risk] Fuzzy retrieval may over-resolve wrong cards -> Mitigation: use thresholds, ambiguity reporting, and never silently choose low-confidence candidates.
- [Risk] Agent loops can become slow -> Mitigation: bound iterations, cache local indexes, and keep network fetch disabled unless explicitly requested.
- [Risk] Coverage checks may be heuristic -> Mitigation: expose coverage report and allow final output to list unresolved gaps.
- [Risk] Prompt-only parsing may miss card names before card search -> Mitigation: combine parser output with retrieval queries over quoted spans, known separators, and scenario noun phrases.
- [Risk] Final JSON answer can fail validation -> Mitigation: add retry/repair or deterministic fallback.

## Migration Plan

1. Add LangGraph dependency and graph module without changing existing CLI/Web defaults.
2. Refactor existing adjudication retrieval operations into reusable graph tools.
3. Add fuzzy card search using current card retrieval components.
4. Add graph state, nodes, conditional edges, coverage report, and trace output.
5. Add opt-in execution path through service/CLI/Web settings.
6. Once tests show parity or better behavior, make the graph path the default adjudication implementation while retaining fallback behavior.

## Open Questions

- Should the first graph path be opt-in via flag, or become default once tests pass?
- What score thresholds should distinguish resolved, ambiguous, and unresolved card candidates?
- Should final-answer JSON repair be implemented immediately or deferred?
- Should tool trace be returned by default or only in debug mode?
