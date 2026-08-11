## Context

The adjudication agent currently uses deterministic scenario parsing and mention slicing to generate many possible card-name queries. It then resolves these queries with normalized name matching and sparse local retrieval. This keeps the system offline-capable and deterministic, but it creates noisy traces and does not behave like an agent loop when mentions are ambiguous, misspelled, or embedded in long natural-language clauses.

LLM support exists, but it currently attempts to parse the whole scenario and can fail with invalid JSON. The next architecture should use LLMs for a narrower task: extracting and selecting candidate card names while keeping all card ids constrained to local retrieval results.

## Goals / Non-Goals

**Goals:**

- Introduce an explicit card-resolution loop inside the adjudication agent.
- Use LLM extraction to propose possible card-name strings without trusting LLM-provided ids.
- Use local card RAG as the only source of candidate card ids and card texts.
- Allow rule-based selection for high-confidence candidates and optional LLM selection only within the retrieved candidate set.
- Retry unresolved or ambiguous mentions with targeted query expansion before evidence retrieval.
- Preserve deterministic fallback behavior when LLM extraction or selection fails.
- Improve traceability by recording extraction source, candidate lists, selection rationale, retry attempts, and warnings.

**Non-Goals:**

- Do not replace the local card database as the source of truth.
- Do not allow LLMs to invent cards, ids, effects, or rulings outside retrieved local candidates.
- Do not implement ygoenv simulation as part of this change.
- Do not redesign FAQ/rules evidence retrieval except where it consumes the resolved-card set.

## Decisions

### Decision: Split card resolution into explicit loop stages

The agent will separate card resolution into:

1. mention extraction
2. local candidate retrieval
3. candidate selection
4. targeted retry for unresolved/ambiguous mentions
5. resolved-card finalization

Rationale: the current flow mixes mention generation and selection in one pass. Separating stages makes retries and trace output meaningful.

Alternative considered: keep current slicing and only tune thresholds. That would reduce some false positives but would not support an agent loop or LLM-assisted disambiguation.

### Decision: LLM extracts strings, not card ids

The LLM extraction prompt will return possible card-name strings and optional context spans, not selected card ids.

Rationale: local `cards.cdb` remains the source of truth, and LLM hallucinated ids are avoided.

Alternative considered: ask LLM to identify exact card ids. Rejected because it would be unreliable and bypass local data.

### Decision: Candidate selection is constrained to local retrieval

Rule-based selection will pick candidates with exact/substring high confidence. Optional LLM selection may choose only among retrieved candidates and must return candidate ids from that set. Unknown ids are rejected and produce warnings/fallback.

Rationale: this preserves correctness and makes LLM behavior auditable.

Alternative considered: allow LLM to ask for arbitrary additional cards. Rejected for this change; targeted retries should still query local card retrieval.

### Decision: Retry unresolved and ambiguous mentions before ruling evidence retrieval

If mentions remain unresolved or ambiguous, the agent will form targeted searches from the original mention, neighboring scenario text, card-name-like substrings, and effect/action terms. The agent will run a bounded number of retries before continuing with warnings.

Rationale: evidence retrieval is significantly better when card ids are resolved, so card resolution deserves its own bounded loop.

Alternative considered: defer ambiguity to final answer generation. Rejected because the evidence retrieval step depends on resolved card ids.

### Decision: Trace card-resolution decisions as first-class agent evidence metadata

The agent trace will include extracted mentions, local candidates, selected ids, selection source, retry count, and warnings.

Rationale: users need to inspect why a card was selected or why a mention stayed unresolved.

## Risks / Trade-offs

- LLM extraction can return malformed JSON -> keep deterministic mention extraction fallback and record a warning.
- LLM selection can overfit to wording -> constrain it to local candidates and validate returned ids.
- More retrieval calls can slow adjudication -> bound retries and top-k, and reuse the existing sparse card retriever.
- More trace data can become noisy -> expose summarized traces by default and keep full details available when trace is requested.
- High-confidence wrong substring matches can still occur -> keep ambiguity margin checks and allow targeted retry when multiple candidates are close.
