## Context

The adjudication agent now resolves card mentions through a pipeline that combines deterministic mention extraction, local card lookup, sparse retrieval, and optional constrained LLM selection. This improves recall, but it creates a correctness risk: sparse retrieval can return a top-ranked card that is unrelated to the mention, and the current selector can treat that relative rank as sufficient evidence to resolve the mention.

The evidence retrieval path has a related issue. Persistent ruling indexes contain enough metadata to distinguish direct FAQ, refer-related FAQ, supplements, analogous FAQ, and OCG rules passages, but the agent currently asks the persistent retriever through one broad evidence call. That makes the retrieval strategy and trace harder to audit than the older runtime `YgocdbRulingIndex` path.

## Goals / Non-Goals

**Goals:**

- Prevent sparse or dense top-1 card retrieval candidates from being resolved without an absolute validity check.
- Introduce explicit card candidate validation outcomes: accepted, ambiguous, rejected, or needs LLM validation.
- Make short mentions and aliases safer by requiring exact names, verified aliases, or validated explanations.
- Preserve direct, related, analogous, supplement, and rules evidence relationships in persistent retrieval.
- Split persistent evidence retrieval into auditable stages with trace metadata showing stage, query, requested card ids, and relationship labels.
- Keep LLM outputs constrained to local candidates and retrieved evidence.

**Non-Goals:**

- Do not implement a full game-state parser in this change.
- Do not allow LLMs to invent card ids, aliases, or evidence ids.
- Do not require a new external card database.
- Do not implement ygoenv simulation.
- Do not redesign vector index storage unless metadata gaps require a small manifest/schema bump.

## Decisions

### Decision: Treat Scenario Parser output as hints

`parse_scenario_with_llm()` output will remain query-expansion input unless later validated. Fields such as `effect_actions`, `replacement_candidates`, and `warnings` must not become authoritative game facts.

Rationale: the parser currently validates JSON shape, not semantic truth. Treating it as a parser of game state would overstate its reliability.

Alternative considered: strengthen the scenario parser into a game-state parser immediately. Rejected because that requires event/chains/constraints modeling and evidence-backed validation, which is larger than this change.

### Decision: Add an absolute card-candidate validity gate

Card resolution will separate candidate generation from candidate validation. Exact normalized name and verified alias matches can resolve directly. Substring matches require coverage constraints. Sparse/dense retrieval candidates cannot resolve solely from rank or relative score; they must pass lexical/alias validation or constrained LLM validation that explains the match.

Rationale: rank means "best among retrieved candidates", not "valid explanation of the mention".

Alternative considered: raise sparse score thresholds. Rejected because it still uses relative ranking and will fail for short or noisy mentions.

### Decision: Short mentions require stricter proof

Mentions below a configured length or token threshold will require exact name, verified alias, or explicit user quoted/known-card context. Sparse/dense retrieval can surface candidates but must leave them ambiguous or unresolved without proof.

Rationale: short mentions such as abbreviations are high-risk and often collide with unrelated card text.

### Decision: Introduce local verified aliases as optional source of truth

The resolver may load a local alias table for common abbreviations and user-maintained names. Alias entries must map to card ids and carry a source such as `manual`, `official_variant`, or `imported_verified`.

Rationale: abbreviations are real user input and should be handled deterministically when known, rather than inferred from sparse card text.

### Decision: Stage persistent evidence retrieval

The agent will retrieve evidence in explicit stages:

1. direct FAQ and supplements for resolved card ids
2. refer-related FAQ for resolved card ids
3. analogous FAQ search using the mechanism query
4. OCG rules search using the mechanism query

Each stage will preserve relationship labels and trace metadata.

Rationale: direct and related rulings have different authority from analogous examples and rules passages. The final answer prompt and tests need that distinction.

Alternative considered: keep one broad hybrid search and infer labels afterward. Rejected because it makes recall and evidence priority less auditable.

## Risks / Trade-offs

- Stricter card validation may leave more mentions unresolved. Mitigation: expose candidate lists and unresolved reasons in trace, and support verified aliases.
- Existing tests that assumed sparse top-1 resolution may need updates. Mitigation: add tests that distinguish candidate generation from resolution.
- Direct/related staged retrieval may add more calls to persistent indexes. Mitigation: use small per-stage top-k values and dedupe evidence by id.
- Alias data can become stale. Mitigation: keep aliases local, explicit, and testable; do not auto-accept LLM-generated aliases.
- Evidence stage ordering can bias final answers. Mitigation: retain all stage labels and let answer-generation priority rules prefer direct/related evidence.

## Migration Plan

- Keep existing card database and ruling index formats initially.
- Add validation metadata to card resolution traces without removing existing fields.
- If alias data is introduced, load it as an optional local JSON artifact and keep behavior unchanged when absent.
- Preserve existing CLI/Web output fields; add relationship stage metadata rather than replacing evidence objects.
- Update batch adjudication test cases to assert expected unresolved states where strict validation rejects weak candidates.

## Open Questions

- What default minimum mention length should trigger strict short-mention handling for CJK and ASCII tokens?
- Should alias files be project-maintained only, or should user-local alias overlays be supported?
- Should constrained LLM validation be enabled by default for ambiguous candidates, or remain opt-in?
- Should direct and related persistent retrieval use separate indexes or metadata filters over the existing index?
