## 1. Data Models and Source Indexes

- [x] 1.1 Add adjudication dataclasses for parsed scenarios, resolved cards, evidence items, reasoning steps, final answers, confidence, warnings, and simulation status
- [x] 1.2 Add ygocdb FAQ and supplement document models with `fid`, primary card id, `refer` card ids, HTML fields, plain-text fields, dates, source URL, and snapshot metadata
- [x] 1.3 Implement ygocdb API response cache loading for `https://ygocdb.com/api/v0/card/{id}?show=all`
- [x] 1.4 Implement FAQ deduplication by `fid` and reverse lookup indexes for primary card ids and `faqs[*].refer` card ids
- [x] 1.5 Implement supplement ingestion as lower-priority card supplement evidence
- [x] 1.6 Implement OCG rules EPUB parsing from `ocg-rule-readthedocs-io-zh-cn-latest.epub` using `nav.xhtml` or `toc.ncx`
- [x] 1.7 Implement rules section indexing with stable `ocgrules:{xhtml_path}#{anchor}` evidence ids and secondary chunking for long sections
- [x] 1.8 Add configuration settings for ygocdb cache path, rules EPUB/index path, and adjudication retrieval limits

## 2. Scenario Parsing and Card Resolution

- [x] 2.1 Implement structured scenario parsing for card mentions, roles, effect activations, objects, replacement candidates, and asked questions
- [x] 2.2 Add an LLM-backed parser prompt with parseable structured output and deterministic parse-error handling
- [x] 2.3 Implement card mention resolution against the local card database with exact, normalized, and ambiguity reporting
- [x] 2.4 Add tests for the provided destruction/replacement scenario and ambiguous card-name scenarios

## 3. Evidence Retrieval

- [x] 3.1 Add exact card-effect evidence generation for all resolved involved cards
- [x] 3.2 Add direct ygocdb FAQ retrieval by primary card id
- [x] 3.3 Add related-card ygocdb FAQ retrieval by `faqs[*].refer` reverse index
- [x] 3.4 Add analogous FAQ retrieval using similar-effect retrieval and mechanism-level query terms
- [x] 3.5 Add supplement evidence retrieval with lower priority than FAQ evidence
- [x] 3.6 Add OCG rules passage retrieval that runs for every adjudication request when the rules index is available
- [x] 3.7 Add evidence deduplication, stable evidence ids, kind labels, source relationship labels, and retrieval warnings
- [x] 3.8 Add tests showing direct FAQ retrieval, refer-based related FAQ retrieval, supplement retrieval, and OCG rules retrieval

## 4. Evidence-Grounded Reasoning

- [x] 4.1 Implement the `RulingAdjudicator` orchestration service from scenario parsing through final response assembly
- [x] 4.2 Add an evidence-constrained LLM reasoning prompt requiring citations to known evidence ids
- [x] 4.3 Add deterministic validation that final reasoning cites only returned evidence ids
- [x] 4.4 Add fallback output when LLM reasoning is unavailable or evidence is insufficient
- [x] 4.5 Add tests for exact evidence, analogous-only evidence, missing evidence, and invalid evidence citation cases

## 5. CLI and Web Entry Points

- [x] 5.1 Add a CLI adjudication command or mode with scenario text, card database path, ruling index path, rules index path, and simulation flag options
- [x] 5.2 Add Web API parsing and validation for adjudication requests
- [x] 5.3 Add Web UI controls for adjudication mode while preserving the existing retrieval UI
- [x] 5.4 Add tests for CLI help, CLI adjudication execution, Web validation, and Web response shape

## 6. Simulation Adapter Boundary

- [x] 6.1 Add a simulator verifier protocol and response type for `not_requested`, `not_available`, `not_encoded`, and observed outcomes
- [x] 6.2 Wire simulation request flags through CLI, Web API, and the adjudication service without requiring `ygoenv`
- [x] 6.3 Add placeholder tests proving document-grounded adjudication still returns when simulation is unavailable
- [x] 6.4 Document follow-up requirements for concrete `ygoenv` scenario encoding and result comparison

## 7. Validation and Documentation

- [x] 7.1 Update README or docs with the adjudication workflow, ygocdb cache/index behavior, EPUB rules ingestion, evidence priority, and limitations
- [x] 7.2 Run `openspec validate --changes add-ai-ruling-adjudication`
- [x] 7.3 Run the full test suite with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

## 8. Post-Implementation Normalization

- [x] 8.1 Decode `datas.category` card effect tags using `strings.conf` `!system 1100+` labels with a correctly ordered fallback map
- [x] 8.2 Add decoded card effect tags to card retrieval documents and adjudication card-effect evidence metadata
- [x] 8.3 Use resolved card effect tags to improve mechanism queries and analogous FAQ retrieval
- [x] 8.4 Download and read `strings.conf` from the YGOPro database source without using the system proxy by default
- [x] 8.5 Change OCG rules EPUB parsing to extract anchor-local sections rather than whole XHTML pages
- [x] 8.6 Change OCG rules chunking to use paragraph-aware chunks with shorter, more complete passages
- [x] 8.7 Add weighted OCG rules search using section title and chapter path
- [x] 8.8 Refactor final adjudication prompt into a testable builder that requires fixed JSON fields
- [x] 8.9 Add JSON answer validation for required fields and known evidence ids
- [x] 8.10 Run focused adjudication tests and the full test suite after the normalization changes
