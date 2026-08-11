## 1. Evidence Context Builder

- [x] 1.1 Add a typed adjudication evidence context builder for planner and coverage critic prompts.
- [x] 1.2 Always include all resolved `card_effect` evidence with full text in planner and critic contexts.
- [x] 1.3 Include resolved-card supplements and current claim-cited evidence ahead of latest low-priority evidence.
- [x] 1.4 Add bounded sections for direct FAQ, related FAQ, analogous FAQ, and rules evidence.
- [x] 1.5 Replace `state.evidence[-N:]` prompt construction with the evidence context builder.
- [x] 1.6 Add tests proving card effects are retained after many FAQ/rules evidence items are added.

## 2. Stable Claim And Issue Graph

- [x] 2.1 Add stable issue/claim id generation based on question, branch, claim type, and semantic role.
- [x] 2.2 Add data structures for issue polarity: affirmative, negative, conditional, and unknown.
- [x] 2.3 Add parser compatibility for existing claim coverage reports while accepting new issue/polarity fields.
- [x] 2.4 Change first `critique_coverage` to initialize the graph and later critiques to update existing graph nodes.
- [x] 2.5 Reject or normalize LLM outputs that reuse an existing id for a different claim meaning.
- [x] 2.6 Add trace serialization for stable graph ids, polarity, dependencies, and evidence checks.

## 3. Coverage Aggregation

- [x] 3.1 Update aggregation so evidence can support a negative issue outcome instead of only contradicting a positive claim.
- [x] 3.2 Implement dependency inference for `case_application` and `hypothetical_branch` nodes from supported parent claims.
- [x] 3.3 Require inference reasons and supported dependency ids before derived application support is accepted.
- [x] 3.4 Preserve hard contradiction handling when evidence supports mutually incompatible outcomes.
- [x] 3.5 Add deterministic gap generation that targets unresolved stable ids.
- [x] 3.6 Add tests for supported positive, supported negative, conditional, unknown, contradicted, and dependency-supported outcomes.

## 4. Coverage Critic Prompt

- [x] 4.1 Rewrite the coverage critic prompt in Chinese around issue graph update semantics.
- [x] 4.2 Instruct the critic to update existing ids rather than regenerate the full graph after the first critique.
- [x] 4.3 Instruct the critic to classify evidence as supporting affirmative, supporting negative, conditional, contradicting, or irrelevant.
- [x] 4.4 Instruct the critic that exact same-scenario FAQ is not mandatory when supported dependencies entail the application.
- [x] 4.5 Include current graph state and evidence context sections in the prompt.
- [x] 4.6 Add prompt snapshot or structure tests for required Chinese instructions and schema fields.

## 5. Planner Loop And Budgets

- [x] 5.1 Exclude `critique_coverage` from ordinary duplicate-tool warnings or include evidence revision in its signature.
- [x] 5.2 Add evidence revision tracking for coverage critique observations.
- [x] 5.3 Add categorized settings for retrieval, critique, submit, and total planner budgets.
- [x] 5.4 Update the agent loop to enforce categorized budgets and report which budget is exhausted.
- [x] 5.5 Ensure at least one submit attempt remains after ordinary retrieve-and-critique cycles under default settings.
- [x] 5.6 Improve fallback warnings to distinguish budget exhaustion, invalid answer synthesis, and insufficient coverage.

## 6. Retrieval Query Policy

- [x] 6.1 Update the planner prompt in Chinese to distinguish FAQ searches from rules searches.
- [x] 6.2 Allow FAQ searches to retain card names, exact effect names, and resolved card id filters.
- [x] 6.3 Require rules searches to use mechanism-focused terms rather than card names as primary terms.
- [x] 6.4 Add validation or warnings for card-name-heavy `search_rules` queries.
- [x] 6.5 Add tests that rules-query rewrites target mechanism terms for failed-destruction and subsequent processing.

## 7. Submit Validation And Answer Generation

- [x] 7.1 Extend `submit_ruling` schema and validation to accept stable issue ids and supported negative outcomes.
- [x] 7.2 Validate that final answers cite stable issue/claim ids from the latest graph.
- [x] 7.3 Allow supported negative conclusions to be submitted with normal confidence when evidence entails the negative outcome.
- [x] 7.4 Require unresolved issues to be listed in uncertainties for low-confidence answers.
- [x] 7.5 Keep evidence relationship validation for card effects, supplements, FAQ, analogous evidence, and rules.

## 8. Tests And Verification

- [x] 8.1 Add unit tests for evidence context retention and ordering.
- [x] 8.2 Add parser tests for stable graph updates, id reuse rejection, and polarity fields.
- [x] 8.3 Add aggregation tests for negative supported conclusions and dependency-supported applications.
- [x] 8.4 Add loop tests for repeated critique after new evidence without duplicate warnings.
- [x] 8.5 Add budget tests for retrieval, critique, submit, and total exhaustion messages.
- [x] 8.6 Add query policy tests for FAQ versus rules retrieval.
- [x] 8.7 Add full-chain regression for the unchained replacement case expecting a supported negative ruling rather than fallback.
- [x] 8.8 Run focused adjudication tests, full test suite, and a real LLM trace smoke test without proxy.
