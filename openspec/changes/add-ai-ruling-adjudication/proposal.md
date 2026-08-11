## Why

Users need answers to concrete Yu-Gi-Oh! ruling scenarios, not only similar card-effect retrieval. The current RAG flow can find card text, but it does not identify all involved cards, gather official and analogous rulings, consult rule documents, or explain a defensible ruling chain.

## What Changes

- Add an AI ruling adjudication workflow for user-provided concrete game scenarios.
- Extract involved card names, effect activations, target/object choices, replacement effects, unresolved questions, and expected answer points from a free-form Chinese scenario.
- Retrieve exact card effects from the local YGOPro card database.
- Retrieve official ruling entries from `https://ygocdb.com/api/v0/card/{id}?show=all`, using `faqs` as ruling Q&A evidence and `supplement` as lower-priority card supplement evidence.
- Use `faqs[*].refer` to index rulings for all referenced similar cards, including entries where one ruling is attached to one card but applies to multiple related cards.
- Retrieve relevant passages from the local `ocg-rule-readthedocs-io-zh-cn-latest.epub` documentation alongside ruling entries instead of treating rules lookup as only a fallback.
- Produce a structured Chinese ruling answer with evidence, inference steps, uncertainty, and unresolved data gaps.
- Leave local `ygoenv` simulation as a later verification phase behind an interface boundary, because scenario setup and replay validation are substantially more complex than document-grounded adjudication.

## Capabilities

### New Capabilities

- `ai-ruling-adjudication`: Covers scenario parsing, card/effect grounding, ruling and rule-document retrieval, evidence-based reasoning, structured adjudication output, and optional future simulation verification.

### Modified Capabilities

- `cli-interface`: Add a user-facing command or mode for adjudication queries without changing existing card retrieval commands.
- `web-ui`: Add a user-facing API/UI path for adjudication queries without changing existing retrieval response compatibility.

## Impact

- Affected modules likely include new adjudication services under `rag_agent/`, card lookup helpers, retrieval/indexing code for non-card documents, CLI routing, Web API routing, and tests.
- New runtime data sources are needed for cached ygocdb ruling entries and parsed OCG rules documentation; these should remain local data and out of git when large or regenerated.
- The DeepSeek-compatible LLM remains optional but is required for best-quality scenario parsing and final reasoning unless deterministic parsers are later added for a narrower subset.
- `ygoenv` integration is explicitly deferred to a later implementation phase and should be designed behind an adapter so document-grounded adjudication can ship first.
