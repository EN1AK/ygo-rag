## 1. Dependencies and State Models

- [x] 1.1 Add LangGraph dependency or optional dependency wiring for the adjudication agent path
- [x] 1.2 Define serializable `AdjudicationAgentState`, card candidate, coverage report, and trace entry models
- [x] 1.3 Add configuration for max graph iterations, card candidate thresholds, and trace/debug output
- [x] 1.4 Add tests for initial graph state creation and state serialization

## 2. Card Resolution Tools

- [x] 2.1 Implement card-name normalization for spaces, middle dots, full-width punctuation, brackets, and separators
- [x] 2.2 Refactor exact and normalized card lookup into a reusable card-resolution tool
- [x] 2.3 Add fuzzy card search using the existing sparse card retriever over card name, effect text, and effect tags
- [x] 2.4 Add candidate scoring and ambiguity thresholds with explicit resolved, ambiguous, and unresolved statuses
- [x] 2.5 Add tests for `破械冥官·篁` resolving to `破械冥官 篁` and for ambiguous fuzzy matches

## 3. Evidence Tool Extraction

- [x] 3.1 Refactor card effect evidence generation into a graph-callable tool
- [x] 3.2 Refactor direct and related ygocdb FAQ retrieval into graph-callable tools with provenance labels
- [x] 3.3 Refactor supplement retrieval into a graph-callable lower-priority evidence tool
- [x] 3.4 Refactor analogous FAQ search into a graph-callable mechanism/effect-tag search tool
- [x] 3.5 Refactor OCG rules search into a graph-callable targeted mechanism search tool
- [x] 3.6 Add tests for each evidence tool's output shape and relationship labels

## 4. LangGraph Nodes and Routing

- [x] 4.1 Implement scenario extraction node using the existing deterministic/LLM parser
- [x] 4.2 Implement card search and candidate selection nodes
- [x] 4.3 Implement card effect, FAQ, analogous FAQ, supplement, and rules retrieval nodes
- [x] 4.4 Implement coverage check node for unresolved cards, ambiguous cards, missing evidence, and hypothetical branch gaps
- [x] 4.5 Implement conditional edges for card resolution, evidence retrieval, retry, and final answer routing
- [x] 4.6 Enforce max iteration limits and record retry reasons in state
- [x] 4.7 Add unit tests for routing decisions and loop termination

## 5. Final Answer and Validation

- [x] 5.1 Adapt the structured adjudication answer prompt to consume graph state, coverage report, and evidence trace summary
- [x] 5.2 Validate final JSON answers against fixed fields and known evidence ids
- [x] 5.3 Add one retry or deterministic fallback when final JSON validation fails
- [x] 5.4 Ensure hypothetical questions produce both actual-case and hypothetical-branch conclusions
- [x] 5.5 Add regression tests for the destruction replacement scenario requiring `faq:24345` or equivalent evidence for the hypothetical branch

## 6. Service, CLI, and Web Integration

- [x] 6.1 Add an adjudication service option to run the LangGraph agent path
- [x] 6.2 Preserve the existing adjudication response fields and add optional coverage/trace data
- [x] 6.3 Add CLI option or configuration flag to select the agent path and trace output
- [x] 6.4 Add Web API support for selecting the agent path and returning coverage/trace data when requested
- [x] 6.5 Add CLI and Web tests for agent-path request/response compatibility

## 7. Validation and Documentation

- [x] 7.1 Update README or docs with the LangGraph adjudication agent flow, fuzzy card resolution, coverage checks, and limitations
- [x] 7.2 Run `openspec validate --changes add-langgraph-adjudication-agent`
- [x] 7.3 Run focused adjudication tests
- [x] 7.4 Run the full test suite with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
