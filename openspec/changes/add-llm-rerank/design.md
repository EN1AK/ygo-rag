## Context

The current retrieval pipeline always builds a sparse retriever, can optionally add Chroma dense retrieval, and can optionally rerank fused candidates with a local `bge-reranker-v2-m3` model. The local reranker improves ordering but adds deployment friction on low-compute machines, especially when GPU memory is limited or Windows native model stacks are unstable.

The project already has a DeepSeek-compatible LLM integration for final answer synthesis. This change uses that existing LLM integration as an optional judge/reranker path. It does not remove local reranking.

## Goals / Non-Goals

**Goals:**

- Add an optional LLM rerank mode that can order fused retrieval candidates without loading the local bge reranker.
- Keep current sparse, dense, hybrid, and local reranker behavior intact.
- Make the rerank provider explicit in CLI/Web/query service behavior.
- Require structured LLM judge output so ranking can be parsed, tested, and audited.
- Keep ordinary tests offline by using fake LLM responses.

**Non-Goals:**

- Do not replace DeepSeek final answer synthesis.
- Do not make LLM rerank the default.
- Do not remove the existing local bge reranker or subprocess reranker.
- Do not introduce a hosted reranker service in this change.
- Do not solve the full "effect similarity intent model" problem; this change only adds an LLM-based ordering option over candidate cards.

## Decisions

### Decision: Add a rerank provider mode instead of overloading `--rerank`

Introduce an explicit rerank provider selection such as `none`, `local`, and `llm`. Existing `--rerank` behavior should continue to mean local bge rerank for compatibility. New CLI/API options can select LLM rerank without implying final answer generation.

Rationale: LLM rerank and final LLM answer are separate operations. A user may want LLM rerank but retrieval-only output, or local rerank plus LLM answer.

Alternatives considered:

- Reuse `--llm` to imply LLM rerank. Rejected because answer synthesis and reranking are different costs and should be independently controlled.
- Replace local reranker with LLM reranker. Rejected because local rerank remains useful when offline or when API cost/latency matters.

### Decision: Treat LLM rerank as a `Reranker`-compatible adapter

LLM rerank should fit the current reranker protocol: accept a query and a candidate list, return reordered candidates with scores and metadata.

Rationale: This keeps `CardRagAgent.retrieve()` mostly stable and preserves current sparse/dense/hybrid flow.

Alternatives considered:

- Put LLM judge logic inside final answer generation. Rejected because ranking should be available even when final answer synthesis is disabled.
- Create a separate post-processing step after `RetrievedCard`. Rejected because it would duplicate candidate conversion and make score semantics less clear.

### Decision: Require structured JSON judge output

The LLM judge prompt should ask for JSON containing candidate card ids, numeric scores, and concise reasons. The implementation should validate that returned ids match input candidates and should fail clearly if the response cannot be parsed.

Rationale: Free-form LLM ranking would be hard to test and unsafe to feed back into deterministic retrieval.

Alternatives considered:

- Ask the LLM to return a ranked prose list. Rejected because parsing is brittle.
- Use function calling/tool calling. Deferred because current DeepSeek integration uses generic `ChatOpenAI`; JSON-only prompting is enough for the first implementation.

### Decision: Limit the LLM judge candidate set

LLM rerank should only receive the fused candidate pool controlled by `rerank_candidates` and should enforce a documented maximum. This limit may be environment-configurable.

Rationale: LLM cost and latency scale with candidate count and card text length.

Alternatives considered:

- Send every retrieved sparse/dense candidate. Rejected due to cost and prompt-length risk.

### Decision: Preserve source card text and judge reason in results

When LLM rerank is used, result metadata should preserve the original source card text and include a judge reason or explanation derived from the structured LLM output.

Rationale: Users need to audit whether LLM judge promoted a card for the intended effect similarity or a superficial keyword match.

## Risks / Trade-offs

- [Risk] LLM judge may produce invalid JSON → Mitigation: validate response, fail with a clear error, and test malformed outputs.
- [Risk] LLM judge may hallucinate card ids or invent candidates → Mitigation: ignore or reject ids not present in input candidates.
- [Risk] LLM rerank increases API latency and token cost → Mitigation: keep it opt-in and limit candidate count.
- [Risk] LLM judge may still over-weight incidental terms such as cost → Mitigation: prompt the judge to score primary effect intent separately from superficial keyword overlap.
- [Risk] Provider selection can confuse users → Mitigation: expose clear CLI/Web labels and keep `--rerank` as local rerank compatibility behavior.

## Migration Plan

1. Add tests for provider selection, LLM rerank parsing, and query behavior with a fake LLM.
2. Add the LLM reranker adapter and integrate it through query service configuration.
3. Add CLI and Web parameters for LLM rerank.
4. Update README after implementation.
5. Keep existing local reranker paths unchanged so rollback is selecting `none` or `local`.

## Open Questions

- Should malformed LLM judge output fail the whole query or fall back to hybrid ranking with a warning?
- What maximum candidate count should be allowed for LLM rerank by default?
- Should LLM judge score include separate fields for primary intent similarity, condition similarity, cost similarity, and effect outcome similarity?
