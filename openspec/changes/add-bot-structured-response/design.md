## Context

The Web API currently returns:

```json
{
  "answer": "...",
  "results": [{ "card_id": 1, "name": "...", "score": 0.1, "source_text": "...", "reason": "..." }],
  "warnings": []
}
```

This is usable for a browser but inconvenient for QQ bot integration because a bot should send one or more bounded messages, ideally one card per message. The bot should not have to parse the long answer text or reconstruct card messages from raw fields.

## Goals / Non-Goals

**Goals:**

- Add a response field that contains card-split message units.
- Keep the existing response fields backward compatible.
- Make every message unit self-contained enough for bot delivery.
- Allow optional maximum message length control.
- Avoid requiring LLM final answer generation for bot output.

**Non-Goals:**

- Do not implement the QQ bot itself.
- Do not add QQ platform-specific SDKs or dependencies.
- Do not change retrieval ranking quality.
- Do not remove or rename existing API response fields.

## Decisions

### Decision: Add a `structured` object instead of replacing `answer`

The API should add a new top-level `structured` object containing a summary and per-card message blocks. Existing `answer`, `results`, and `warnings` remain unchanged.

Rationale: This avoids breaking the current Web UI and any existing API consumers.

### Decision: Use card-level message blocks

Each card result should produce one block with:

- `type`: `card`
- `index`
- `card_id`
- `name`
- `score`
- `text`: ready-to-send message text
- `fields`: structured source fields used to build the text
- `truncated`: whether text was shortened to fit the requested limit

Rationale: Bots can send `structured.blocks[*].text` directly, or build custom messages from `fields`.

### Decision: Keep splitting deterministic and local

Structured card blocks should be generated from retrieved result fields and not require another LLM call.

Rationale: Bot output should be available in low-cost retrieval-only mode and should not introduce additional API latency.

### Decision: Optional request flag controls structured output

Callers should be able to request structured output explicitly, while the implementation may also include it by default if low-risk. If an explicit flag is added, use a clear name such as `structured=true`.

Rationale: Explicit flags make bot integration predictable and leave room for future response formats.

## Risks / Trade-offs

- [Risk] Message text may still exceed QQ limits if source text is long → Mitigation: support a max length and report `truncated`.
- [Risk] Overly terse truncation may hide important card text → Mitigation: keep raw fields available separately from the display text.
- [Risk] Multiple response formats can confuse callers → Mitigation: keep one additive `structured` object and document it.

## Open Questions

- What exact QQ message length limit should be used as the default?
- Should the first block be a summary/header message, or should callers build their own header?
- Should source text truncation preserve full text in `fields.source_text` or truncate both display text and field text?
