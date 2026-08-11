## Why

The adjudication agent currently treats coverage as sufficient once card names are resolved and some evidence exists. This is too weak for ruling work because it does not verify that retrieved evidence is relevant to each required conclusion or that the evidence actually entails the answer.

This change introduces claim-level coverage so the agent exposes and audits the proof structure before final submission, reducing premature answers and case-specific retrieval behavior.

## What Changes

- Add a Claim Coverage Report produced by `critique_coverage` instead of a single loose sufficient/insufficient judgment.
- Require coverage analysis to generate required claims for the scenario, including main questions and hypothetical branches.
- Distinguish evidence relevance from evidence entailment for each claim.
- Add five claim statuses: `supported`, `partially_supported`, `unsupported`, `contradicted`, and `unknown`.
- Add claim dependency graph output so final conclusions can be traced through required subclaims.
- Let code aggregate the Claim Coverage Report into a deterministic submit gate rather than trusting the LLM to declare coverage sufficient.
- Require `submit_ruling` answers to cite supported claim ids in addition to evidence ids.
- Let the planner continue targeted retrieval when claim gaps remain, and prevent final submission when required claims are unsupported, contradicted, or unknown.
- Preserve the existing agent loop and retrieval tools; this change strengthens coverage and answer validation rather than replacing retrieval.

## Capabilities

### New Capabilities

- `claim-coverage-report`: Defines claim extraction, evidence relevance checks, evidence entailment checks, claim statuses, dependency graph output, gap reporting, and deterministic coverage aggregation for adjudication.

### Modified Capabilities

- `answer-generation`: Final adjudication answers must be grounded in supported claim ids and valid evidence ids, and must account for unsupported, contradicted, unknown, or partially supported claims.

## Impact

- Affected code:
  - `rag_agent/adjudication_agent.py`
  - `rag_agent/adjudication.py`
  - `rag_agent/adjudication_service.py`
  - CLI/Web trace serialization for claim coverage reports and claim-linked final answers
- Affected tests:
  - claim coverage report parsing and validation
  - relevance versus entailment status tests
  - claim dependency graph tests
  - planner loop tests where gaps trigger targeted retrieval
  - submit validation tests for unsupported, contradicted, unknown, and partially supported claims
  - compatibility tests for existing deterministic fallback and trace output
- No new external data source is required. Existing card database, YGOCDB FAQ cache, persistent FAQ/rules indexes, and LLM configuration remain reusable.
