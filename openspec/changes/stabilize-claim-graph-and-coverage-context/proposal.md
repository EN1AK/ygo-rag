## Why

The adjudication agent now avoids unsupported final submissions, but real LLM traces show it can still fail to answer because coverage critique loses early card-effect evidence, regenerates unstable claim ids, treats contradiction as mere missing support, and exhausts the loop budget before submission.

This change stabilizes the proof structure and evidence context so the agent can move from "not enough evidence" to a supported positive or negative ruling conclusion.

## What Changes

- Build a deterministic coverage evidence context that always preserves resolved card effects, card supplements, claim-cited evidence, high-priority FAQ, and latest retrieval results instead of passing only the last N evidence items.
- Introduce a stable claim graph lifecycle: first critique creates claim identities, later critiques update status/evidence/gaps without renumbering or rewriting existing claims.
- Add conclusion polarity or issue-answer modeling so evidence can support a negative answer, not only make a positive claim `unsupported`.
- Allow `case_application` claims to be supported by dependency inference from supported `card_text`, `rule_principle`, and `ruling_fact` claims when no exact same-scenario FAQ exists.
- Exclude `critique_coverage` from ordinary duplicate-tool warnings, or deduplicate it by evidence revision rather than static tool arguments.
- Separate adjudication loop budgets for retrieval, critique, and submit attempts so coverage review does not consume all opportunities to answer.
- Improve planner rules for retrieval query generation: FAQ tools may keep card names, while rules search must abstract card-specific facts into mechanism-focused queries.
- Improve fallback and budget-exhausted messaging so users can distinguish invalid synthesis, insufficient coverage, and loop budget exhaustion.

## Capabilities

### New Capabilities

- `claim-graph-coverage`: Stable issue/claim graph creation, update, evidence-context construction, polarity handling, dependency inference, and deterministic coverage aggregation for adjudication.

### Modified Capabilities

- `answer-generation`: Agent final answers must cite stable claim or issue ids and may submit supported negative conclusions when evidence contradicts a positive proposition.
- `retrieval-ranking`: Adjudication rules retrieval must use mechanism-focused queries distinct from card-name-heavy FAQ queries.
- `runtime-configuration`: Agent loop budget settings must distinguish retrieval, critique, submit, and total iteration limits.

## Impact

- Affected code:
  - `rag_agent/adjudication_agent.py`
  - `rag_agent/adjudication.py`
  - `rag_agent/adjudication_service.py`
  - `rag_agent/config.py`
  - `rag_agent/ruling_index.py`
- Affected tests:
  - claim graph stability and update tests
  - evidence context retention tests
  - positive/negative conclusion polarity tests
  - dependency inference coverage tests
  - duplicate critique and loop budget tests
  - rules-query abstraction tests
  - real LLM trace regression tests for the unchained replacement case
- No new external data source is required. Existing card text, YGOCDB FAQ/supplement cache, persistent ruling indexes, and OCG rules EPUB remain the evidence sources.
