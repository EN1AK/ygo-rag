## 1. Claim Coverage Data Model

- [x] 1.1 Add claim coverage dataclasses or typed structures for claims, evidence checks, gaps, dependencies, report metadata, and aggregation results.
- [x] 1.2 Define enums for claim status, claim type, evidence relevance, and evidence entailment.
- [x] 1.3 Implement strict parser/validator for Claim Coverage Report JSON, including unknown claim dependency rejection.
- [x] 1.4 Add serialization helpers for agent trace, CLI output, and Web output.

## 2. Coverage Critique Tool

- [x] 2.1 Replace the current loose `critique_coverage` observation with a structured Claim Coverage Report while preserving legacy coverage fields for compatibility.
- [x] 2.2 Build a coverage critique prompt that asks for required claims, relevance checks, entailment checks, gaps, contradictions, and suggested queries.
- [x] 2.3 Ensure the coverage LLM does not return authoritative `sufficient`; code must compute submit readiness from the report.
- [x] 2.4 Include main-case and hypothetical-branch claims separately when questions contain hypothetical branches.
- [x] 2.5 Add claim dependency graph validation and expose dependencies in trace output.

## 3. Deterministic Submit Gate

- [x] 3.1 Implement deterministic coverage aggregation from claim statuses, claim dependencies, and evidence checks.
- [x] 3.2 Block normal-confidence submission when decisive claims are `unsupported`, `contradicted`, or `unknown`.
- [x] 3.3 Require `partially_supported` decisive claims to be answered with explicit uncertainty or low confidence.
- [x] 3.4 Require branch-specific supported claims for hypothetical conclusions.
- [x] 3.5 Feed claim gaps and suggested queries back into planner prompts for targeted retrieval.

## 4. Final Answer Validation

- [x] 4.1 Extend `submit_ruling` answer schema to allow or require claim ids in reasoning entries.
- [x] 4.2 Validate that cited claim ids exist in the latest accepted Claim Coverage Report.
- [x] 4.3 Validate that cited claims are supported or that uncertainty rules are satisfied.
- [x] 4.4 Validate that cited evidence relationship labels in `evidence_use` match the evidence pool.
- [x] 4.5 Reject overconfident answers that rely only on analogous evidence for decisive claims without stating the analogy limitation.

## 5. Planner Loop Integration

- [x] 5.1 Update planner prompt to include the latest Claim Coverage Report, claim gaps, contradictions, and suggested queries.
- [x] 5.2 Adjust loop policy so `coverage.sufficient` is derived from claim aggregation, not from evidence existence.
- [x] 5.3 Allow continued retrieval after a critique only when the planner targets a reported claim gap or contradiction.
- [x] 5.4 Prevent repeated searches that do not target unresolved claim gaps.
- [x] 5.5 Preserve deterministic fallback when claim coverage cannot be parsed or no accepted submission is possible.

## 6. CLI And Web Trace Output

- [x] 6.1 Add Claim Coverage Report summaries to CLI trace output.
- [x] 6.2 Add Claim Coverage Report rendering data to Web responses.
- [x] 6.3 Ensure non-trace responses remain compact and backwards compatible.

## 7. Tests And Verification

- [x] 7.1 Add parser tests for valid reports, malformed reports, invalid statuses, missing fields, and unknown dependency ids.
- [x] 7.2 Add coverage aggregation tests for all five claim statuses.
- [x] 7.3 Add relevance-versus-entailment tests where evidence is relevant but does not support the claim.
- [x] 7.4 Add contradiction tests where evidence supports the opposite conclusion.
- [x] 7.5 Add hypothetical-branch tests requiring branch-specific claims.
- [x] 7.6 Add submit validation tests for unknown claim ids, unsupported claims, contradicted claims, partially supported claims, and evidence relationship mismatch.
- [x] 7.7 Add planner-loop tests where claim gaps trigger targeted FAQ/rules retrieval.
- [x] 7.8 Add real-case regression tests for the adjudication cases under `scripts/adjudication_cases`.
- [x] 7.9 Run focused adjudication tests, full test suite, and at least one real LLM trace smoke test.
