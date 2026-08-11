## 1. Ruling Document Models

- [x] 1.1 Add normalized ruling document dataclasses for FAQ, supplement, and OCG rules passages
- [x] 1.2 Convert ygocdb FAQ payloads into `ygocdb:faq:{fid}` retrieval documents with primary/refer card metadata
- [x] 1.3 Convert ygocdb supplements into `ygocdb:supplement:{card_id}` retrieval documents with provenance metadata
- [x] 1.4 Convert parsed OCG rule passages into retrieval documents with chapter/XHTML/anchor metadata
- [x] 1.5 Include card names and `strings.conf` effect tags in indexed FAQ/supplement text and metadata
- [x] 1.6 Add unit tests for document conversion, stable evidence ids, and metadata preservation

## 2. Persistent Index Build and Manifest

- [x] 2.1 Add configuration for FAQ/supplement index path, rules index path, collection names, and manifest paths
- [x] 2.2 Implement Chroma build service for FAQ/supplement ruling documents
- [x] 2.3 Implement Chroma build service for OCG rules documents
- [x] 2.4 Write index manifests with schema version, source paths, source counts, document count, embedding model, collection name, and timestamp
- [x] 2.5 Validate manifests before dense retrieval and report stale/missing/incompatible index warnings
- [x] 2.6 Add reset, limit, and batch-size support for ruling index builds
- [x] 2.7 Add tests for index build, reset behavior, manifest write, and manifest validation

## 3. Ruling Retrieval Pipeline

- [x] 3.1 Implement sparse retrieval over ruling documents using Chinese/Japanese/ASCII tokens, card names, FAQ ids, effect tags, and mechanism terms
- [x] 3.2 Implement dense vector retrieval from persistent FAQ/supplement and rules collections
- [x] 3.3 Implement reciprocal-rank fusion and deterministic tie breaking for ruling evidence candidates
- [x] 3.4 Preserve relationship labeling for direct_card, related_card, analogous, card_supplement, and rule_document evidence
- [x] 3.5 Convert retrieved ruling candidates back into `EvidenceItem`-compatible objects
- [x] 3.6 Add tests for direct FAQ, related FAQ, analogous FAQ, supplement, and rules retrieval
- [x] 3.7 Add regression test that the destruction replacement scenario retrieves FAQ `24345` or equivalent evidence

## 4. Reranking

- [x] 4.1 Reuse local bge reranker for ruling evidence candidates
- [x] 4.2 Add LLM judge rerank prompt and parser for ruling evidence candidates
- [x] 4.3 Reject LLM judge evidence ids outside the retrieved candidate set
- [x] 4.4 Preserve unscored candidates deterministically after LLM-scored candidates
- [x] 4.5 Fall back to fused/local order with warnings when LLM judge rerank fails
- [x] 4.6 Add tests for local rerank, LLM rerank, unknown id rejection, malformed output fallback, and timeout fallback

## 5. Agent and Service Integration

- [x] 5.1 Add ruling index service objects to adjudication service construction
- [x] 5.2 Update LangGraph evidence tools to prefer persistent FAQ/supplement and rules indexes
- [x] 5.3 Keep runtime ygocdb and EPUB parsing fallback when persistent indexes are missing
- [x] 5.4 Record persistent-index usage and fallback reasons in agent trace
- [x] 5.5 Ensure coverage checks run over reranked persistent evidence
- [x] 5.6 Add tests for agent persistent-index path, fallback path, and trace metadata

## 6. CLI and Diagnostics

- [x] 6.1 Add CLI command to build FAQ/supplement and OCG rules indexes
- [x] 6.2 Add CLI options for FAQ-only, rules-only, reset, limit, batch-size, and index paths
- [x] 6.3 Add CLI command to inspect ruling index manifests and collection document counts
- [x] 6.4 Add CLI command to query persistent ruling evidence directly
- [x] 6.5 Add CLI options for semantic retrieval, local rerank, LLM rerank, top-k, and JSON/text output
- [x] 6.6 Add CLI tests for build help, inspect missing indexes, query output shape, and LLM rerank selection

## 7. Documentation and Validation

- [x] 7.1 Update README with ruling index build/query/adjudication workflows
- [x] 7.2 Document recommended workflow: download ygocdb cache, build ruling indexes, run adjudication agent
- [x] 7.3 Document fallback behavior and limitations when persistent indexes are stale or missing
- [x] 7.4 Run `openspec validate --changes add-persistent-ruling-rag-index`
- [x] 7.5 Run focused ruling index and adjudication agent tests
- [x] 7.6 Run the full test suite with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
