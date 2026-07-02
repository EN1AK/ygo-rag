## 1. Filterable metadata

- [x] 1.1 Add tests that retrieval document metadata exposes scalar/boolean Chroma-filterable fields such as `is_xyz`, `is_link`, `rank`, `decoded_level`, and `attribute_name`.
- [x] 1.2 Implement filterable metadata fields while preserving existing raw and decoded metadata fields.

## 2. Dense metadata filters

- [x] 2.1 Add tests for translating structured query filters into Chroma `where` filter dictionaries.
- [x] 2.2 Add tests that `ChromaVectorStore.search()` forwards metadata filters to Chroma.
- [x] 2.3 Implement optional metadata filter support in the dense retriever interface and Chroma wrapper.
- [x] 2.4 Add fallback behavior and warnings when metadata-filtered dense search fails.

## 3. Wider filtered recall

- [x] 3.1 Add tests proving structured-filtered sparse retrieval keeps candidates ranked below final `top_k` but within the first 100 eligible for fusion/rerank.
- [x] 3.2 Implement a minimum recall pool of 100 for sparse and dense retrieval when structured filters are present.
- [x] 3.3 Preserve existing unfiltered retrieval candidate behavior.

## 4. Documentation and validation

- [x] 4.1 Update README to explain that dense metadata filtering requires rebuilding Chroma with `build-index --reset`.
- [x] 4.2 Run `pytest`.
- [x] 4.3 Run `openspec validate --changes improve-filtered-recall`.
