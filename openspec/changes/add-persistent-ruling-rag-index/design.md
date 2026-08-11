## Context

The current ruling adjudication implementation can read ygocdb FAQ payloads from `data/ygocdb/*.json`, parse the local OCG rules EPUB into passages, and retrieve evidence for the LangGraph adjudication agent. The weak point is that FAQ/rules evidence is still mostly runtime-built: the service may scan JSON payloads, parse the EPUB, build in-memory indexes, and perform sparse matching during an adjudication run.

The existing card RAG path already has the shape we want: card records are converted into retrieval documents, embedded into persistent Chroma storage, fused with sparse retrieval, optionally reranked locally, and optionally judged by an LLM reranker. This change applies the same retrieval architecture to ruling evidence while preserving stable evidence ids for final citation validation.

## Goals / Non-Goals

**Goals:**

- Persist ygocdb FAQ, ygocdb supplement, and OCG rules passage indexes.
- Use vector retrieval for semantic ruling questions and effect-processing interactions.
- Keep sparse retrieval for exact FAQ/card/rule id, card names, effect tags, and mechanism terms.
- Use hybrid fusion and reranking in the same style as card RAG.
- Allow LLM judge rerank over retrieved FAQ/rules candidates before final answer synthesis.
- Preserve direct/related/analogous provenance and evidence ids.
- Make the adjudication agent prefer persistent indexes but fall back to runtime parsing when indexes are absent.
- Provide CLI build/inspect/query workflows for ruling indexes.

**Non-Goals:**

- Do not fetch live ygocdb data as part of normal index querying.
- Do not replace raw `data/ygocdb/*.json` cache files; the index is derived from them.
- Do not implement ygoenv simulation in this change.
- Do not make LLM rerank mandatory; local/offline retrieval must still work.
- Do not change the card RAG index format unless a shared helper needs a backwards-compatible extraction.

## Decisions

### Decision: Create dedicated ruling document models

Create normalized document models for:

- `faq`: one document per ygocdb FAQ `fid`
- `supplement`: one document per card supplement
- `rule_passage`: one document per parsed OCG rules passage

Each document should include:

- stable `evidence_id`
- `kind`
- `source`
- `title`
- searchable text
- `card_ids`
- `primary_card_ids`
- `refer_card_ids`
- `relationship_hints`
- effect tags for referenced cards when available
- source URL, snapshot timestamp, EPUB path/chapter/anchor metadata

Alternative considered: store only raw text in Chroma metadata. That would make citation validation and relationship labeling brittle because the adjudication agent would need to reconstruct provenance after retrieval.

### Decision: Use separate Chroma collections for FAQ/supplement and rules

Use dedicated collections such as `ruling_faqs` and `ocg_rules` under the configured Chroma persistence directory, or under a separate configured ruling index path if needed. Keeping FAQ and rules separate lets the agent tune top-k, filters, and prompts differently.

Alternative considered: one combined evidence collection. That simplifies setup but makes source-specific filtering and scoring harder.

### Decision: Hybrid retrieval mirrors card RAG

Ruling retrieval should have the same phases as card query retrieval:

1. Sparse retrieval over normalized Chinese/Japanese/ASCII terms, card names, effect tags, FAQ ids, and mechanism terms.
2. Dense retrieval from Chroma when the persistent collection is complete and compatible.
3. Reciprocal-rank fusion of sparse and dense candidates.
4. Optional local rerank with the existing bge reranker.
5. Optional LLM judge rerank over the fused/reranked candidate set.

Alternative considered: vector-only retrieval. FAQ/rules data contains many exact card names, ids, and short mechanism terms where sparse recall is more reliable.

### Decision: Build indexes from local sources only

Index build commands should consume already downloaded local data:

- `data/ygocdb/*.json`
- configured `cards.cdb` for card names/effect tags
- configured OCG rules EPUB

The command may validate missing source files and report counts, but normal index building should not silently hit the network. Fetching/updating ygocdb cache remains a separate workflow.

Alternative considered: fetch missing card payloads during index build. That makes builds harder to reproduce and conflicts with the current no-proxy-by-default download expectation.

### Decision: Store and validate an index manifest

Each index should have a manifest containing schema version, source counts, source paths, source file mtimes or hashes when practical, embedding model, embedding device, collection names, and document counts. Query code should warn or skip dense retrieval when the manifest is absent, incompatible, or clearly stale.

Alternative considered: rely only on Chroma collection count. Counts alone cannot distinguish a FAQ index built from stale ygocdb JSON or a different embedding model.

### Decision: Agent evidence tools use persistent indexes first

The LangGraph agent should call persistent ruling retrieval tools for:

- direct/related FAQ by card id
- analogous FAQ by mechanism/effect tags/scenario
- supplement evidence
- OCG rules passages

If the persistent index is missing or stale and runtime sources exist, the agent may fall back to current runtime `YgocdbRulingIndex` and `RuleIndex` behavior with a warning/trace entry.

Alternative considered: require indexes before adjudication. That would improve performance predictability but would break existing no-index workflows.

### Decision: LLM rerank remains bounded and evidence-only

LLM judge rerank should receive only retrieved evidence candidates with ids, titles, snippets, relationships, source, and retrieval scores. It must not introduce new evidence ids. Unknown ids are rejected and omitted. On timeout or malformed output, the retrieval order falls back deterministically with a warning.

Alternative considered: let the LLM generate new search queries or cite outside evidence. That is more agentic but weakens citation validation and testability.

## Risks / Trade-offs

- [Risk] Index build can be slow or resource-heavy -> Mitigation: batch builds, progress output, reset/limit options, and manifest checks.
- [Risk] Chroma metadata may not fit large FAQ text -> Mitigation: keep large text in page content and store compact metadata only.
- [Risk] Stale indexes can produce incorrect retrieval -> Mitigation: manifest validation and clear warnings.
- [Risk] Dense retrieval can bury exact FAQ matches -> Mitigation: always keep sparse recall and relationship filters in the fused candidate set.
- [Risk] LLM rerank can reorder away key direct FAQs -> Mitigation: preserve relationship priority as a score feature and cap rerank to retrieved candidates only.
- [Risk] Japanese FAQ text and Chinese scenario terms may mismatch semantically -> Mitigation: include card names, effect tags, mechanism expansions, and related-card metadata in indexed text.

## Migration Plan

1. Add ruling document model and source adapters for ygocdb FAQ/supplement and OCG rules passages.
2. Add persistent Chroma build and manifest writing for FAQ/supplement and rules collections.
3. Add sparse and dense ruling retrievers plus hybrid fusion.
4. Add optional local rerank and optional LLM judge rerank for ruling evidence.
5. Add CLI build/inspect/query commands.
6. Update adjudication service and LangGraph evidence tools to prefer persistent indexes.
7. Keep runtime parsing fallback and warnings until indexes are built.
8. Validate with the destruction replacement scenario, requiring FAQ `24345` or equivalent evidence.

Rollback is straightforward: disable the persistent ruling index settings or omit index paths; the agent should fall back to runtime parsing.

## Open Questions

- Should FAQ/supplement and rules live in the existing `CHROMA_PERSIST_DIR` or a separate default such as `data/ruling_chroma`?
- Should direct/related card id filtering happen before dense retrieval, after dense retrieval, or both?
- Should the Web UI expose a manual ruling-index build/status page, or is CLI-only sufficient for now?
- Should index manifests hash all JSON files or use mtime/count for speed?
