## Context

The adjudication agent now runs as a tool loop where the LLM planner can resolve cards, search FAQ/rules evidence, critique coverage, and submit a final ruling. The current coverage model is too shallow: it mostly checks whether card mentions are resolved and whether some evidence exists. A real ruling answer needs a proof structure: each conclusion depends on claims, each claim needs relevant evidence, and relevant evidence may still fail to entail the claim.

The recent real LLM tests show the failure mode clearly. The agent can parse cards, retrieve plausible FAQ evidence, and submit valid JSON, but it may still draw the wrong conclusion because coverage does not audit whether the evidence supports or contradicts the submitted claims.

## Goals / Non-Goals

**Goals:**

- Replace loose coverage with claim-level coverage that enumerates required claims for each question and hypothetical branch.
- Separate evidence relevance from evidence entailment.
- Support five claim states: `supported`, `partially_supported`, `unsupported`, `contradicted`, and `unknown`.
- Add a claim dependency graph so final conclusions can be traced to required subclaims.
- Let the LLM produce structured coverage analysis, but let code compute the submit gate from that structure.
- Require final answers to cite supported claim ids and the evidence ids behind those claims.
- Preserve the agent loop model where the planner uses gaps to generate targeted follow-up retrieval queries.

**Non-Goals:**

- Do not implement a full game-state simulator.
- Do not require ygoenv simulation in this change.
- Do not assume one fixed retrieval query per scenario.
- Do not make the LLM's `sufficient` judgment authoritative.
- Do not require perfect legal proof of every answer; the first implementation should provide auditable gaps and block obviously unsupported final submissions.

## Decisions

### Decision: `critique_coverage` returns a Claim Coverage Report

`critique_coverage` will stop acting as a thin missing-cards/missing-evidence checker. It will produce a structured report containing:

- `required_claims`
- `evidence_checks`
- `claim_dependencies`
- `gaps`
- `contradictions`
- `suggested_queries`

Rationale: the planner needs actionable gaps rather than a boolean. A claim report also creates a stable artifact for tests and UI inspection.

Alternative considered: ask the planner directly whether evidence is sufficient. Rejected because it recreates the current problem; a boolean is too easy for the LLM to overstate.

### Decision: Relevance and entailment are separate fields

Each evidence check will include both relevance and entailment:

- `relevance`: whether the evidence is about the same card, mechanism, timing, branch, or rule principle.
- `entailment`: whether the evidence supports, partially supports, contradicts, or does not establish the claim.

Rationale: many retrieved rulings are topically similar but do not prove the conclusion. Treating topical similarity as support is the main failure mode.

Alternative considered: one `status` field per evidence item. Rejected because it hides the difference between irrelevant evidence and relevant but non-entailing evidence.

### Decision: Code computes the submit gate

The LLM may assign claim statuses, but code will compute whether submission is allowed:

- Required claims with `unsupported`, `contradicted`, or `unknown` block normal-confidence submission.
- `partially_supported` claims block submission unless they are explicitly marked non-decisive or the final answer reports uncertainty/low confidence.
- At least one decisive claim must be supported for each answer branch.
- Hypothetical branches must have their own claims and may not borrow the main-case claim unless dependencies explicitly justify it.

Rationale: the LLM should explain the proof; code should enforce minimum standards.

### Decision: Final answers cite claims, not only evidence

`submit_ruling` will require reasoning entries to cite `claim_id` values from the latest accepted Claim Coverage Report. Evidence ids remain required, but they are routed through supported claims.

Rationale: evidence-only citation lets the LLM cite a valid FAQ id while making an unsupported inference. Claim ids let validators check whether the cited evidence has already been assessed for relevance and entailment.

### Decision: Claim dependency graph is first-class but lightweight

The graph will be represented as adjacency data inside the report:

```json
{
  "claim_id": "c5",
  "depends_on": ["c1", "c3"]
}
```

Rationale: conclusions in rulings often depend on card text, rule principles, and case application claims. A lightweight graph is enough for validation and display without introducing a graph database.

### Decision: The planner can continue retrieval from gaps

When the report contains gaps, the planner prompt will include `suggested_queries` and the claim ids they target. The planner may call FAQ/rules/card tools with mechanism-specific queries.

Rationale: this keeps the agent loop flexible while grounding the next searches in explicit missing claims.

## Risks / Trade-offs

- Claim reports can be verbose -> Keep prompt summaries compact and expose only recent/high-signal evidence to the coverage LLM.
- The LLM may mislabel claim status -> Require evidence ids, relevance reasons, entailment reasons, and contradictions; add deterministic schema and submit-gate checks.
- Over-strict gating may block useful answers -> Allow low-confidence answers only when unsupported/unknown claims are explicitly represented as uncertainties.
- More loop steps may increase latency -> Use max-step controls, duplicate-query checks, and gap-targeted searches.
- The claim schema may evolve -> Version the report schema in trace metadata so future changes can be migrated.

## Migration Plan

1. Add dataclasses/types and parsers for claim coverage reports without changing the existing fallback answer path.
2. Update `critique_coverage` to produce structured claim reports while preserving legacy missing-card/evidence fields for compatibility.
3. Add deterministic coverage aggregation from claim reports.
4. Update planner prompts to use claim gaps and suggested queries.
5. Update `submit_ruling` validation to require claim ids and supported claim status.
6. Update CLI/Web trace output to show claim coverage reports.
7. Run compatibility tests and real LLM case tests; tune prompts only through generic claim schema rules, not sample-specific terms.

## Open Questions

- Should claim extraction be done by `critique_coverage` only, or should there be a separate `draft_claims` tool?
- Should direct/related FAQ evidence be required for decisive ruling claims when available, or can rules + analogous FAQ be sufficient?
- How should low-confidence answers be represented in the current answer JSON schema?
- Should the system store claim coverage reports in output JSON by default or only when trace is requested?
