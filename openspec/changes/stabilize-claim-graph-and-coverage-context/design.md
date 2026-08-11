## Context

The adjudication agent already uses a tool loop with card resolution, FAQ/rules retrieval, coverage critique, and structured final submission. The previous claim coverage work added claim statuses, evidence checks, and deterministic submit gating. Real LLM traces now show the next failure mode: the gate prevents unsupported answers, but the agent can get stuck because the proof object and the evidence context are unstable.

The concrete trace for the unchained replacement case shows:

- resolved card effects exist in state, but card-effect evidence can be omitted from planner/critic prompts because the prompt uses the last N evidence items
- repeated `critique_coverage` regenerates a different claim set and reuses `c1`, `c2`, etc. for different meanings
- evidence that supports the negative ruling is treated as "does not entail" a positive claim rather than support for a negative conclusion
- repeated critique calls are marked duplicate because their tool signature has no evidence revision
- the fixed total step budget is consumed by retrieve/critique cycles before any accepted `submit_ruling`
- rules search queries can overfit card names and retrieve broad summon-limit passages instead of mechanism rules

## Goals / Non-Goals

**Goals:**

- Preserve all resolved `card_effect` evidence and other high-priority evidence in planner and coverage critic prompts.
- Create a stable claim or issue graph that survives multiple critique rounds.
- Represent conclusion polarity so evidence can support "no" as well as "yes".
- Allow application conclusions to be inferred from supported dependencies without requiring an exact same-scenario FAQ.
- Treat `critique_coverage` as an expected repeated review step keyed to evidence revisions, not as a duplicate ordinary tool call.
- Separate retrieval, critique, submit, and total budgets so the loop can still submit after gathering evidence.
- Make rules search queries mechanism-focused while leaving FAQ queries card-specific.
- Keep prompts in Chinese by default.

**Non-Goals:**

- Do not implement ygoenv simulation.
- Do not add a new external data source.
- Do not weaken citation or evidence validation.
- Do not make the LLM's coverage sufficiency boolean authoritative.
- Do not require every ruling to have a direct same-scenario FAQ.

## Decisions

### Decision: Build a deterministic coverage evidence context

Planner and critic prompts will use an evidence context builder rather than `state.evidence[-N:]`.

The context builder will always include:

- every resolved card's `card_effect` evidence, with full text
- direct card supplements for resolved cards when present
- evidence cited by the current claim graph
- direct and related FAQ for unresolved claims
- latest retrieval results
- a bounded number of analogous FAQ/rule passages by relevance

Rationale: card text is foundational evidence and must not disappear because later FAQ/rule retrieval filled the prompt window.

Alternative considered: increase the `[-N:]` window. Rejected because it only delays the same failure and wastes tokens.

### Decision: Use stable claim graph lifecycle

The first coverage critique creates the graph. Later critiques update existing nodes by stable identity and may append new nodes, but they may not renumber or rewrite the graph wholesale.

Stable ids should be derived by code from question, branch, claim type, and semantic role where possible, for example:

```text
q1.main.card_text.shenlong_replacement_effect
q1.main.application.can_shenlong_apply_replacement
q2.hypothetical.application.can_huang_special_summon
```

Rationale: stable ids let the planner target gaps over multiple rounds and let final answers cite a durable proof object.

Alternative considered: ask the LLM to preserve `c1`, `c2`, etc. Rejected because traces show the LLM readily reuses short ids for different claims.

### Decision: Prefer issue-answer polarity over single positive claims

Coverage should model adjudication questions as issues with supported answers:

```json
{
  "issue_id": "q1.main.can_shenlong_apply",
  "issue": "Can Shenlong apply its replacement effect?",
  "supported_answer": "no",
  "status": "supported",
  "supporting_evidence_ids": ["ygocdb:faq:24336"]
}
```

The implementation may keep claim objects internally, but must represent whether the supported conclusion is affirmative, negative, conditional, or unknown.

Rationale: a positive claim becoming unsupported is not enough. Ruling systems need to distinguish "no evidence" from "evidence supports the opposite conclusion."

Alternative considered: rely only on `contradicted` status. Rejected as incomplete because final answers still need a supported negative conclusion to cite.

### Decision: Dependency inference supports case application

`case_application` and `hypothetical_branch` conclusions may become supported through dependencies when:

- parent card text claims are supported by `card_effect`
- parent ruling or rule principle claims are supported by FAQ/supplement/rules evidence
- the application node has an explicit inference reason
- no stronger contradiction remains unresolved

Rationale: exact same-scenario FAQ is often unavailable. The agent must support rule-based adjudication without hallucinating.

Alternative considered: require direct FAQ for every application claim. Rejected because it makes coverage permanently insufficient for combinatorial card interactions.

### Decision: Critique calls are keyed by evidence revision

`critique_coverage` will either be excluded from ordinary duplicate detection or its signature will include a deterministic evidence revision/hash. A repeated critique after new evidence is not a duplicate.

Rationale: the natural loop is retrieve -> critique -> retrieve -> critique.

Alternative considered: keep duplicate warnings but ignore them. Rejected because warnings pollute trace and can mislead planner and users.

### Decision: Split loop budgets

The loop will track separate limits:

- retrieval tool calls
- coverage critique calls
- submit attempts
- total planner turns

When budget is near exhaustion, the planner should prefer a final critique and then a low-confidence `submit_ruling` if no hard contradiction prevents it.

Rationale: total step budgets should not be consumed entirely by internal review.

Alternative considered: only raise the current max iteration default. Rejected because it does not distinguish useful retrieval from repeated review.

### Decision: Rules queries are mechanism-focused

The planner prompt and tool validators should distinguish FAQ and rules retrieval:

- FAQ searches may include card names and exact effect phrases.
- Rules searches should abstract card-specific facts into mechanism questions and avoid card names as core terms.

Rationale: rules documents are organized by mechanics, not card names. Card-name-heavy rules queries retrieve broad unrelated sections such as summon-limit pages.

Alternative considered: let embedding handle it. Rejected because real traces show card names and broad terms dominate the embedding match.

## Risks / Trade-offs

- More structured claim/issue data increases implementation complexity -> keep schema versioned and add parser tests for migrations.
- Polarity modeling can overfit yes/no questions -> support `conditional` and `unknown` answers explicitly.
- Dependency inference can become too permissive -> require explicit parent ids, inference reason, and relationship-aware evidence checks.
- Larger evidence context can exceed token budgets -> use typed sections and full text only for essential card effects/supplements; keep FAQ/rules snippets bounded.
- More loop budget settings add configuration surface -> provide conservative defaults and preserve existing environment behavior where possible.

## Migration Plan

1. Introduce evidence context builder and use it in planner and critic prompts.
2. Add stable claim/issue ids and parser support while preserving old report fields for compatibility.
3. Add polarity fields and deterministic aggregation rules for positive, negative, conditional, and unknown answers.
4. Update planner and submit validation to cite stable issue/claim ids.
5. Split duplicate detection and loop budgets.
6. Add query validation/guidance for FAQ versus rules tools.
7. Run focused parser/aggregation tests, full suite, and real LLM trace regression for the unchained replacement case.

## Open Questions

- Should the public response expose both issue ids and claim ids, or only issue ids?
- Should low-confidence final submission be forced when evidence remains insufficient, or should the fallback remain deterministic for no-answer cases?
- How strict should automatic rules-query rejection be when a card name also doubles as a common mechanic term?
