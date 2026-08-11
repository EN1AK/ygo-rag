## 1. Card Resolution Validity

- [x] 1.1 Locate current mention extraction, `search_card_candidates()`, `_resolve_card_mention()`, and `select_card_resolution()` call sites and document the existing trace fields that must remain backward compatible.
- [x] 1.2 Introduce candidate validation metadata for card id, name, score, retrieval source, match reason, normalized-name comparison, validation status, validation reason, and selection source.
- [x] 1.3 Implement deterministic acceptance for exact normalized card-name matches and verified alias matches.
- [x] 1.4 Implement substring acceptance only when configured length and coverage constraints are satisfied.
- [x] 1.5 Add short-mention handling so sparse or dense retrieval candidates cannot resolve short mentions without exact-name or verified-alias proof.
- [x] 1.6 Change sparse and dense card retrieval paths so top-ranked candidates are returned as candidates only and cannot directly set `resolved`.
- [x] 1.7 Constrain LLM card selection to local candidate ids and reject or downgrade selections that lack a compatibility explanation.
- [x] 1.8 Add unresolved and ambiguous trace reasons for rejected candidates and validation failures.

## 2. Persistent Evidence Retrieval

- [x] 2.1 Audit persistent ruling document metadata for FAQ primary card ids, refer card ids, supplements, rules, document ids, and current relationship inference.
- [x] 2.2 Split `_persistent_ruling_evidence()` into explicit direct FAQ, related FAQ, supplement, analogous FAQ, and rules retrieval stages.
- [x] 2.3 Filter direct and related stages by resolved card ids and skip those stages when no resolved card ids exist.
- [x] 2.4 Preserve `direct_card`, `related_card`, `card_supplement`, `rule_document`, and `analogous` relationship labels from persistent metadata.
- [x] 2.5 Update deduplication so stronger direct or related relationships are retained when the same evidence also appears in analogous results.
- [x] 2.6 Add retrieval stage trace metadata with stage name, requested card ids, query summary, returned evidence ids, and relationship labels.

## 3. Ranking And Prompt Integration

- [x] 3.1 Update ranking and reranking code so card retrieval score is ranking evidence only and never substitutes for card-resolution validity.
- [x] 3.2 Ensure persistent ruling ranking computes relationship labels before final ranking and does not overwrite stronger labels with `analogous`.
- [x] 3.3 Update final adjudication prompt construction so every evidence item includes relationship and stage metadata.
- [x] 3.4 Keep answer-generation priority as direct FAQ, related FAQ, supplement and card text, analogous FAQ, then rules.

## 4. Tests And Validation

- [x] 4.1 Add card-resolution tests where an unrelated sparse top-1 candidate, including the `雷破` false-positive case, remains unresolved or ambiguous.
- [x] 4.2 Add card-resolution tests for exact names, verified aliases, substring coverage, short mentions without aliases, and invalid LLM selections.
- [x] 4.3 Add persistent retrieval tests proving direct FAQ, refer-related FAQ, supplement, analogous FAQ, and rules evidence keep distinct relationship labels.
- [x] 4.4 Add deduplication tests proving direct or related labels survive when the same evidence is returned by an analogous stage.
- [x] 4.5 Update end-to-end adjudication tests to assert resolved card ids, unresolved reasons, evidence ids, relationships, and retrieval stage trace.
- [x] 4.6 Run the focused adjudication, retrieval, and full-chain test suites and record any remaining test gaps.
