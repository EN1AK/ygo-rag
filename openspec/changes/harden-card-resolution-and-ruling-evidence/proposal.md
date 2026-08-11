## Why

The adjudication agent currently risks accepting unrelated sparse card retrieval results as resolved cards, because card selection is based mostly on relative rank and score rather than whether a candidate can actually explain the mention. Persistent ruling retrieval also needs clearer evidence relationship semantics so direct card FAQ, refer-related FAQ, analogous FAQ, supplements, and rules remain distinguishable after moving to indexed retrieval.

## What Changes

- Add absolute validity gates to card resolution so sparse top-1 candidates cannot be resolved solely because they rank first.
- Require card candidates to prove lexical, alias, or constrained semantic compatibility with the original mention before becoming `resolved`.
- Treat sparse/dense card retrieval as candidate generation unless a separate validation step establishes that the candidate explains the mention.
- Add explicit handling for short or ambiguous mentions, such as common abbreviations, where exact name or verified alias evidence is required.
- Split persistent ruling evidence retrieval into direct, related, analogous, supplement, and rules stages.
- Preserve evidence relationship labels from persistent indexes instead of flattening retrieval results into a single analogous search.
- Keep Scenario Parser output as query-expansion hints unless later validated by card resolution or evidence coverage.

## Capabilities

### New Capabilities

- `card-resolution-validity`: Defines absolute validity requirements for resolving card-name mentions from exact names, aliases, substrings, and retrieval candidates.
- `staged-ruling-evidence`: Defines staged evidence retrieval and relationship preservation for persistent FAQ, supplement, analogous, and rules evidence.

### Modified Capabilities

- `retrieval-ranking`: Card and ruling retrieval ranking must distinguish candidate generation from validated selection and preserve evidence relationship semantics.

## Impact

- Affected code:
  - `rag_agent/adjudication_agent.py`
  - `rag_agent/ruling_index.py`
  - `rag_agent/retrieval/*`
  - CLI/Web trace and test helpers that summarize resolved cards and evidence relationships
- Affected tests:
  - card resolution tests for sparse false positives, short mentions, aliases, ambiguity, and unresolved states
  - persistent ruling retrieval tests for direct, related, analogous, supplement, and rules relationship labels
  - end-to-end adjudication cases that assert card ids, evidence ids, and coverage
- No new external data source is required, though verified alias data may be introduced as a local project artifact.
