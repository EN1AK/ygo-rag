## 1. Planner And State Foundations

- [x] 1.1 Add agent-loop state fields for planner notes, hypotheses, tool history, evidence pool, unresolved/ambiguous card state, submit attempts, warnings, and step count.
- [x] 1.2 Define allowed planner action names and JSON argument schemas for all adjudication tools.
- [x] 1.3 Implement strict planner response parsing for one action per iteration with validation errors for malformed JSON, unknown tools, and invalid arguments.
- [x] 1.4 Build the planner prompt from scenario text, questions, resolved card state, evidence summary, coverage gaps, recent tool history, and allowed tool schemas.
- [x] 1.5 Add duplicate tool-call signature tracking and loop-limit handling.

## 2. Tool Layer

- [x] 2.1 Implement a tool dispatcher that executes validated actions and returns structured observations without trusting planner facts.
- [x] 2.2 Wrap existing card resolution as `resolve_cards`, preserving absolute validity gates and candidate validation metadata.
- [x] 2.3 Add `search_card` as candidate-only card search that never marks cards resolved by itself.
- [x] 2.4 Add `search_direct_faq` and `search_related_faq` tools filtered by resolved card ids and preserving relationship labels.
- [x] 2.5 Add `search_analogous_faq` accepting planner-generated query text and preserving stronger direct/related labels when metadata proves them.
- [x] 2.6 Add `search_rules` accepting planner-generated query text and returning `rule_document` evidence.
- [x] 2.7 Add `critique_coverage` to summarize unresolved cards, ambiguous cards, unanswered questions, weak evidence, and suggested next searches.
- [x] 2.8 Add `submit_ruling` to validate final JSON structure, evidence citations, unresolved-card uncertainty handling, and confidence constraints.

## 3. Agent Loop Integration

- [x] 3.1 Replace the fixed extract/resolve/retrieve/coverage path with an `agent_plan -> execute_tool -> agent_plan` loop behind a migration-safe entry point.
- [x] 3.2 Route accepted `submit_ruling` output into the existing `AdjudicationResponse` fields.
- [x] 3.3 Preserve the existing fixed workflow or deterministic fallback for absent LLM, repeated invalid actions, step limit exhaustion, and invalid final submissions.
- [x] 3.4 Expose planner decisions, sanitized tool arguments, observations, added evidence ids, validation errors, and final submission status in agent trace.
- [x] 3.5 Update CLI and Web response summaries so users can inspect the agent loop steps and generated retrieval queries.

## 4. Answer Generation And Validation

- [x] 4.1 Update final answer validation so agent-loop answers are accepted only through `submit_ruling`.
- [x] 4.2 Ensure every cited evidence id in submitted answers exists in the current evidence pool.
- [x] 4.3 Reject or downgrade submissions that ignore unresolved or ambiguous card mentions.
- [x] 4.4 Return fallback answers with warnings when no valid submission is accepted.

## 5. Tests And Verification

- [x] 5.1 Add planner parser tests for valid actions, malformed JSON, unknown tool names, invalid arguments, and duplicate-call warnings.
- [x] 5.2 Add tool tests for card resolution, candidate-only card search, direct FAQ, related FAQ, analogous FAQ, rules search, coverage critique, and submit validation.
- [x] 5.3 Add agent-loop tests where the planner recovers an unresolved card before searching evidence.
- [x] 5.4 Add agent-loop tests where the planner generates different rules queries for different mechanism gaps.
- [x] 5.5 Add final submission tests for unknown evidence ids, unresolved-card uncertainty, invalid JSON, and accepted strict JSON.
- [x] 5.6 Add compatibility tests proving existing CLI/Web adjudication entry points still return responses and traces.
- [x] 5.7 Run focused adjudication, ruling retrieval, CLI/Web, and OpenSpec validation suites and record remaining gaps.
