## 1. Test Coverage

- [x] 1.1 Add unit tests for rerank provider selection covering none, local, and LLM provider modes
- [x] 1.2 Add unit tests for LLM rerank prompt construction using candidate ids, names, source text, and scores
- [x] 1.3 Add unit tests for parsing valid LLM judge JSON into card id, score, and reason records
- [x] 1.4 Add unit tests for rejecting unknown LLM judge card ids
- [x] 1.5 Add unit tests for preserving unscored candidates after scored candidates in deterministic order
- [x] 1.6 Add error-path tests for malformed LLM judge output

## 2. Core Rerank Implementation

- [x] 2.1 Add rerank provider representation for none, local, and LLM modes
- [x] 2.2 Implement an LLM reranker adapter compatible with the existing reranker protocol
- [x] 2.3 Implement structured LLM judge prompt generation
- [x] 2.4 Implement structured LLM judge response parsing and validation
- [x] 2.5 Preserve LLM judge reasons in returned candidate metadata
- [x] 2.6 Enforce an LLM rerank candidate limit before sending candidates to the LLM

## 3. Query Service Integration

- [x] 3.1 Extend query request settings to carry rerank provider selection without breaking existing callers
- [x] 3.2 Keep existing `rerank=True` behavior mapped to local bge rerank
- [x] 3.3 Configure LLM rerank independently from final LLM answer generation
- [x] 3.4 Ensure missing DeepSeek credentials produce a clear LLM rerank error
- [x] 3.5 Ensure retrieval-only final answer formatting still works after LLM rerank

## 4. CLI Integration

- [x] 4.1 Add CLI option for selecting LLM rerank
- [x] 4.2 Keep existing `--rerank` flag behavior compatible with local bge rerank
- [x] 4.3 Update CLI help tests to verify the LLM rerank option is listed
- [x] 4.4 Add CLI smoke or unit coverage for LLM rerank selection without live network calls

## 5. Web UI Integration

- [x] 5.1 Add Web UI control for LLM rerank selection separate from final LLM answer generation
- [x] 5.2 Parse LLM rerank selection in `POST /api/query`
- [x] 5.3 Render LLM judge reasons in candidate result cards
- [x] 5.4 Add Web API tests for LLM rerank request parsing

## 6. Documentation and Verification

- [x] 6.1 Update README with LLM rerank usage, cost/latency notes, and credential requirements
- [x] 6.2 Run `openspec validate --change add-llm-rerank`
- [x] 6.3 Run the full test suite with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- [x] 6.4 Manually verify that local rerank behavior still works when LLM rerank is not selected
- [x] 6.5 Add timeout configuration, progress output, and fallback warning for LLM rerank failures
