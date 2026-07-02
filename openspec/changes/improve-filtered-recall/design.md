## Context

The structured query filter change makes queries like `效果是除外对手墓地卡的四星超量怪兽` correctly identify 214 rank 4 XYZ monster candidates. However, retrieval still loses relevant cards because:

- sparse retrieval returns only the final `top_k` candidates from the filtered set;
- dense retrieval queries the full Chroma collection and then filters by allowed card id, so relevant filtered cards outside full-collection top-k are never seen.

The concrete failure case is `No.80 狂装霸王 狂想战曲王`, which matches the structured filter and ranks around 37 in sparse retrieval for the effect query, but default `top_k=10` prevents it from reaching hybrid/rerank.

## Goals / Non-Goals

**Goals:**

- Ensure structured-filtered retrieval uses at least 100 candidates per retrieval source before fusion/rerank.
- Push structured filters into Chroma dense retrieval when the index has compatible filterable metadata.
- Keep unfiltered retrieval behavior unchanged.
- Preserve existing CLI/API flags and response shape, while allowing diagnostics/warnings for degraded dense filtering.
- Make index rebuild requirements explicit.

**Non-Goals:**

- Add a new vector database.
- Add a new reranker.
- Change DeepSeek prompt behavior.
- Solve all effect-intent ranking issues; this change is recall-focused.

## Decisions

### Use 100 as the minimum filtered recall pool

When structured filters are present, sparse retrieval SHALL request at least 100 candidates, bounded by the number of matching filtered cards. Dense retrieval SHALL also request at least 100 candidates from the filtered dense search.

Rationale: this fixes observed misses where relevant cards are ranked below top 10 but inside a moderate candidate pool. It keeps latency bounded for local use.

Alternative considered: use all filtered cards. This maximizes recall but can make reranking and hybrid fusion unnecessarily expensive for broad filters.

### Store scalar filter metadata in Chroma

Chroma metadata filters operate best on scalar values. The index SHALL store fields such as:

- `card_kind`
- `attribute_name`
- `decoded_level`
- `rank`
- `link_rating`
- boolean monster subtype fields such as `is_xyz`, `is_fusion`, `is_synchro`, `is_link`, `is_ritual`, `is_effect`, `is_normal`

Rationale: current `monster_types` is a comma-separated string, which is not suitable for exact metadata filtering. Boolean fields make Chroma `where` filters straightforward.

Alternative considered: filter by an allow-list of card ids. Chroma metadata filters do not scale cleanly with large `$in` lists and older indexes may not have card-id filter metadata in the expected shape.

### Translate structured filters to Chroma where clauses

The dense retriever SHALL accept an optional metadata filter and pass it to Chroma `similarity_search_with_score(..., filter=where)`.

Example for rank 4 XYZ monsters:

```json
{
  "$and": [
    {"card_kind": {"$eq": "monster"}},
    {"is_xyz": {"$eq": true}},
    {"rank": {"$eq": 4}}
  ]
}
```

Rationale: dense semantic ranking should happen inside the filtered subset, not after full-collection retrieval.

### Fall back safely for stale indexes

Existing Chroma indexes may not include new metadata fields. If filtered dense search fails because metadata fields are missing or unsupported, the system SHALL fall back to the current dense post-filter path and report a warning.

Rationale: this avoids hard failures during rollout while still making the rebuild path clear.

## Risks / Trade-offs

- Stale Chroma metadata produces no dense matches → Mitigation: warn and document that `build-index --reset` is required for true dense metadata filtering.
- Larger sparse pool can introduce weaker candidates → Mitigation: hybrid/rerank still controls final top-k; this change only improves recall.
- Chroma filter syntax differs across versions → Mitigation: cover the wrapper behavior with tests and keep fallback logic.
- Boolean subtype fields duplicate `monster_types` → Mitigation: duplication is intentional for vector database filtering.

## Migration Plan

1. Add scalar/boolean decoded metadata to retrieval documents.
2. Add Chroma metadata filter construction and vector-store support.
3. Increase structured-filtered sparse/dense candidate pools to at least 100.
4. Add tests around the `No.80`-style sparse rank > 10 case and Chroma filter forwarding.
5. Update README to instruct users to rebuild Chroma with `--reset` for dense metadata filtering.

Rollback: keep the metadata fields harmless and disable filter passing / candidate expansion if needed.

## Open Questions

- Should the minimum filtered recall size be configurable by environment variable later?
- Should stale-index detection be explicit via an index schema version instead of catching dense filter failures?
