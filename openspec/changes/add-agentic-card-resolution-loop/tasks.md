## 1. Card Resolution Models

- [x] 1.1 Add structured models for extracted card-name mentions, local card candidates, selection decisions, retry attempts, and card-resolution trace summaries.
- [x] 1.2 Add serialization helpers so card-resolution loop state can appear in agent trace and API responses.
- [x] 1.3 Add settings for card-resolution top-k, retry limit, LLM extraction toggle, and optional LLM selection toggle.

## 2. LLM Extraction and Selection

- [x] 2.1 Implement a Chinese structured prompt for extracting possible card-name strings and context spans from adjudication scenarios.
- [x] 2.2 Parse and validate extraction output; reject card ids or unexpected fields from the extraction result.
- [x] 2.3 Implement a constrained LLM selection prompt for ambiguous local candidates.
- [x] 2.4 Validate selection output so only candidate ids present in the local retrieval set can be accepted.
- [x] 2.5 Add deterministic fallback behavior and warnings for malformed extraction or selection output.

## 3. Local Candidate Retrieval

- [x] 3.1 Refactor existing `search_card_candidates()` usage into a candidate provider suitable for agent loop calls.
- [x] 3.2 Ensure each returned candidate includes card id, name, score, source, match reason, and card text needed for LLM disambiguation.
- [x] 3.3 Preserve high-confidence exact and normalized substring selection without requiring LLM selection.
- [x] 3.4 Preserve low-confidence candidate lists without forcing weak matches to resolve.

## 4. Agent Loop Integration

- [x] 4.1 Replace the current one-pass `_node_resolve_cards` behavior with extraction, retrieval, selection, retry, and finalization stages.
- [x] 4.2 Add targeted retry query generation from mention text, neighboring scenario text, and effect/action terms.
- [x] 4.3 Ensure persistent FAQ/rules retrieval consumes only the final resolved-card ids from the card-resolution loop.
- [x] 4.4 Update coverage reporting to distinguish unresolved mentions, ambiguous mentions, and resolved cards after retries.
- [x] 4.5 Add concise card-resolution summary into the final adjudication answer prompt.

## 5. Tests and Validation

- [x] 5.1 Add unit tests for LLM card-name extraction parsing and malformed-output fallback.
- [x] 5.2 Add unit tests for constrained LLM selection rejecting unknown card ids.
- [x] 5.3 Add unit tests for targeted retry resolving a mention that first pass marks unresolved or ambiguous.
- [x] 5.4 Add agent integration tests proving persistent ruling retrieval uses final resolved card ids.
- [x] 5.5 Add regression tests for the sample scenario containing 破械冥官·篁, 破械焰魔天·阎摩, and 完美电子多元驱动蛇·神龙.
- [x] 5.6 Run the relevant pytest suite and `openspec validate --changes add-agentic-card-resolution-loop`.

## 6. Documentation and Observability

- [x] 6.1 Document the new card-resolution loop and its fallback behavior in developer-facing docs or comments where appropriate.
- [x] 6.2 Update CLI/Web trace output expectations if card-resolution trace fields are exposed.
- [x] 6.3 Document recommended settings for top-k and retry limits.
