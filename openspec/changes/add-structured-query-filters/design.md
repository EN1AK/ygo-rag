## Context

The project currently loads raw YGOPro `cards.cdb` fields into `Card` and places those raw values in retrieval metadata. Retrieval uses sparse lexical scoring, optional Chroma dense search, hybrid fusion, and optional reranking. Structural phrases in user queries are not interpreted as constraints, so terms such as `四星超量怪兽` may not reliably recall the intended card set.

The target behavior is a two-stage interpretation:

1. parse card-structure constraints from the user query;
2. use those constraints to prefilter candidate cards before ranking by effect text similarity.

The public `EN1AK/yugioh-ccb` project can be used as a reference for `cards.cdb` bitmask meanings and field decoding. The implementation should keep the decoding local to this project and covered by tests.

## Goals / Non-Goals

**Goals:**

- Decode the subset of CDB metadata needed for query filters from already-loaded fields.
- Parse common Chinese card-structure constraints from natural-language queries.
- Apply parsed filters consistently for CLI and Web/API query execution.
- Keep sparse-only mode, semantic mode, local rerank, and LLM rerank compatible.
- Expose parsed filters and filter diagnostics in API/structured responses so bot integrations can reason about result segmentation and failures.

**Non-Goals:**

- Full natural-language understanding of all Yu-Gi-Oh! rules text.
- Full legality/banlist/format filtering.
- Replacing rerankers or LLM answer generation.
- Adding Qdrant or a hosted vector service.
- Supporting every possible OCG/TCG naming variant in the first implementation.

## Decisions

### Decode CDB fields into a small internal metadata model

Add a local metadata decoder that maps raw `datas.type`, `datas.attribute`, `datas.race`, and `datas.level` into normalized fields such as:

- `card_kind`: `monster`, `spell`, `trap`
- `monster_types`: set-like list containing values such as `normal`, `effect`, `fusion`, `ritual`, `synchro`, `xyz`, `pendulum`, `link`
- `attribute`: normalized value such as `dark`, `light`, `earth`, `water`, `fire`, `wind`, `divine`
- `race`: normalized monster race where supported
- `level`, `rank`, `link_rating`: numeric values derived from raw level/type

Rationale: retrieval code should not need to know CDB bitmask constants. A normalized model also makes tests and API output stable.

Alternative considered: match raw bitmask values directly in query parsing. This is faster to write but spreads CDB knowledge across retrieval and API code.

### Parse filters with deterministic rules first

Implement a deterministic parser for high-signal structural phrases:

- attributes: `暗属性`, `光属性`, etc.
- extra deck types: `超量`, `XYZ`, `融合`, `同调`, `连接`
- monster categories: `仪式`, `效果`, `通常`
- broad card kinds: `怪兽`, `魔法`, `陷阱`
- numeric constraints: `四星`, `4星`, `四阶`, `4阶`, `Rank 4`, `LINK-2`, `连接2`

Rationale: these phrases are finite, testable, and should not require an LLM call. LLM rerank can remain optional and should not be responsible for recall.

Alternative considered: use DeepSeek to parse the whole query into filters. This would improve flexibility but makes base retrieval dependent on network/API availability and introduces harder-to-test failures.

### Keep effect retrieval query separate from filter query

The parser should return both:

- `filters`: normalized structural constraints
- `effect_query`: the remaining text used for sparse/dense retrieval when available

For the example `效果是除外或者回收对手墓地的卡的四星超量怪兽`, the intended parsed form is:

```json
{
  "effect_query": "除外或者回收对手墓地的卡",
  "filters": {
    "card_kind": "monster",
    "monster_types": ["xyz"],
    "rank": 4
  }
}
```

Rationale: keeping structural terms out of the effect query reduces accidental lexical matches on generic terms such as `怪兽` or `四星`.

Alternative considered: append filters to the query string and rely on ranking. That preserves the current pipeline but does not solve recall and ranking mismatch reliably.

### Prefilter before every ranking source

When filters are present, build sparse and dense retrievers over the filtered candidate set, or pass a filter/candidate allow-list to the retrievers. Rerankers only receive the resulting filtered candidates.

Rationale: LLM rerank and local rerank cannot recover candidates that were not retrieved. Filtering before ranking gives structural constraints recall authority.

Alternative considered: filter only after dense/sparse retrieval. This can discard all top candidates and still miss relevant cards outside the original top-k.

### Surface diagnostics without making filtering fatal

Responses should expose:

- parsed filters
- normalized effect query
- candidate counts before and after filtering
- warnings when parsed filters produce zero candidates and the system falls back or returns no results

Rationale: users and bot integrations need to understand why a query returned few or no cards.

Default behavior should preserve existing retrieval when no filters are parsed. For parsed filters that match zero cards, implementation should prefer an explicit warning and deterministic behavior over silently ignoring filters.

## Risks / Trade-offs

- CDB bitmask mistakes → Mitigation: use tests with known synthetic and real-world examples, and compare constants against `EN1AK/yugioh-ccb` as a reference.
- Query parser overmatches effect text as filters → Mitigation: start with conservative phrase patterns and expose parsed filters in output.
- Chroma metadata compatibility breaks existing indexes → Mitigation: detect missing decoded metadata and either rebuild guidance/warning or filter dense results by card id after retrieval from a filtered card set.
- Filtering can return no candidates for ambiguous user wording → Mitigation: include warnings and preserve enough diagnostics for CLI/API users to adjust the query.
- Dynamic retriever rebuilding may add per-query overhead → Mitigation: keep filtered sparse retriever construction simple initially; optimize with cached indexes only if tests or profiling show a real issue.

## Migration Plan

1. Add decoder and parser with unit tests.
2. Add filtering to sparse-only retrieval and query-service output.
3. Extend dense retrieval compatibility so semantic search honors the same candidate filter.
4. Update CLI/API/Web output and structured response.
5. Update docs after behavior is verified.

Rollback is straightforward: disable parser invocation or ignore parsed filters while leaving raw retrieval unchanged.

## Open Questions

- Should parsed filters that match zero cards fall back to unfiltered retrieval, or return zero results with a warning?
- Should `四星超量怪兽` be interpreted leniently as rank 4 XYZ, even though official wording is usually `四阶`?
- Which race terms should be supported in the first implementation?
- Should reference-card expansion happen before or after structured filter parsing when the query contains both a card name and explicit constraints?
