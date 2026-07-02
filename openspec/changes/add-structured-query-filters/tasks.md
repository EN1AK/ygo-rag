## 1. Metadata decoding

- [x] 1.1 Add tests for decoding CDB type, attribute, race, and level-derived values from synthetic card rows.
- [x] 1.2 Implement local CDB metadata constants and decoder for card kind, monster types, attribute, race, level, rank, and link rating.
- [x] 1.3 Add decoded metadata to card document metadata without removing existing raw CDB fields.

## 2. Structured query parsing

- [x] 2.1 Add tests for parsing attributes, card kinds, monster subtypes, rank/level/link numeric phrases, and no-filter queries.
- [x] 2.2 Implement deterministic Chinese structured query parser returning `filters`, `effect_query`, and parse diagnostics.
- [x] 2.3 Cover the target query `效果是除外或者回收对手墓地的卡的四星超量怪兽` as rank 4 XYZ with effect query `除外或者回收对手墓地`.

## 3. Retrieval integration

- [x] 3.1 Add tests that structured filters prefilter sparse retrieval candidates before scoring.
- [x] 3.2 Add tests that semantic retrieval and rerank providers receive only candidates matching parsed filters.
- [x] 3.3 Implement candidate filtering in `CardRagAgent.retrieve()` while preserving existing behavior when no filters are parsed.
- [x] 3.4 Add warnings and diagnostics for zero-match or degraded filter application paths.

## 4. Service and API output

- [x] 4.1 Extend query response data structures with parsed filters, normalized effect query, candidate counts, and filter warnings.
- [x] 4.2 Update Web API serialization to include structured filter diagnostics in top-level JSON and bot `structured` output.
- [x] 4.3 Update CLI output to include structured filter diagnostics when present without breaking existing retrieval-only output.
- [x] 4.4 Update Web UI rendering to show parsed filters and filter warnings near results.

## 5. Validation and documentation

- [x] 5.1 Add or update tests for CLI, query service, Web API, and structured bot response behavior.
- [x] 5.2 Update README usage examples for structured filters after implementation behavior is verified.
- [x] 5.3 Run `pytest` and `openspec validate --changes add-structured-query-filters`.
