## 1. Tests

- [x] 1.1 Add unit tests for building structured card blocks from query results
- [x] 1.2 Add tests for structured block truncation and `truncated` status
- [x] 1.3 Add Web API tests verifying `structured` exists alongside existing response fields
- [x] 1.4 Add Web API tests for request-controlled maximum structured block length

## 2. Structured Response Implementation

- [x] 2.1 Add structured response dataclasses or builder functions in the query/API layer
- [x] 2.2 Generate one card block per result with ready-to-send text and source fields
- [x] 2.3 Implement deterministic text truncation for card blocks
- [x] 2.4 Include structured output in `query_response_to_dict`

## 3. API and Documentation

- [x] 3.1 Parse optional structured response parameters from `POST /api/query`
- [x] 3.2 Update README with bot API response examples
- [x] 3.3 Run `openspec validate --changes add-bot-structured-response`
- [x] 3.4 Run the full test suite with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
