## Context

The current application answers card-effect similarity queries from a local `cards.cdb` database with sparse, dense, reranker, and optional LLM answer generation. That is enough to find relevant card text, but it is not enough to answer a concrete ruling scenario such as whether a destruction replacement effect can apply and whether another effect can continue resolving.

Ruling adjudication needs multiple evidence classes:

- Exact effects for every card mentioned in the scenario.
- Official ruling entries from ygocdb `faqs` for the mentioned cards.
- Related-card ruling entries discovered through `faqs[*].refer`, including entries where a ruling is attached to one representative card and lists other cards as equivalent examples.
- Lower-priority ygocdb `supplement` entries for card-specific supplemental facts.
- OCG rule-document passages from the local `ocg-rule-readthedocs-io-zh-cn-latest.epub`, such as effect resolution, replacement effects, destruction, optional effects, and special summon conditions.
- Later, an executable simulator check through local `ygoenv`, when the scenario can be encoded reliably.

## Goals / Non-Goals

**Goals:**

- Provide a feasible implementation path for AI adjudication that is auditable and testable.
- Separate scenario parsing, evidence retrieval, reasoning, and optional simulation.
- Prefer official rulings and local rule documents as grounded evidence over free-form LLM memory.
- Return a structured answer that shows involved cards, evidence, reasoning steps, conclusion, confidence, and remaining uncertainty.
- Allow CLI and Web users to run adjudication without breaking the existing card retrieval workflow.

**Non-Goals:**

- Do not claim full judge-level correctness for every Yu-Gi-Oh! corner case in the first implementation.
- Do not require `ygoenv` simulation for the first document-grounded adjudication release.
- Do not scrape HTML pages at query time as the primary path; adjudication should work from cached ygocdb API responses and local indexed rule data where possible.
- Do not let the LLM invent card text, ruling entries, or rule passages not present in retrieved evidence.

## Decisions

### Decision: Build a dedicated adjudication pipeline

Add a service such as `RulingAdjudicator` instead of overloading `CardRagAgent.query()`. The pipeline should produce a typed response:

- `scenario`: original user text and parsed structured scenario.
- `cards`: mentioned cards with card ids and effect text.
- `evidence`: card effects, official rulings, analogous rulings, rule-document passages, and optional simulation observations.
- `reasoning`: ordered steps, each linked to evidence ids.
- `answer`: direct Chinese answer to each user question.
- `confidence`: high/medium/low with uncertainty reasons.
- `warnings`: missing card names, missing ruling data, parse ambiguity, or simulator unavailability.

Alternative considered: prompt the existing LLM answer generator with top-k card retrieval results. This is lower effort but not acceptable because the LLM would have no reliable way to distinguish official rulings, analogous rulings, rule documents, and assumptions.

### Decision: Treat scenario parsing as structured extraction

Use an LLM structured-output prompt or parser to extract:

- Card mentions and aliases.
- Controller/turn-player/opponent roles.
- Zones and public state.
- Activated effects, selected targets, objects to destroy, replacement candidates, and asked questions.
- Uncertain phrases that need confirmation.

Then resolve card mentions against the local card database using exact match, normalized match, and ambiguity reporting. The card resolver must return all plausible matches when the name is ambiguous instead of silently choosing one.

Alternative considered: regex-only extraction. It can catch exact card names but will miss roles, effect nesting, and asked subquestions in natural Chinese scenarios.

### Decision: Use ygocdb API as the canonical ruling source

Use `https://ygocdb.com/api/v0/card/{id}?show=all` as the source for card ruling data. The API provides `faqs` and `supplement` fields that should be normalized into evidence documents.

For `faqs`, store:

- `ruling_id`: `ygocdb:faq:{fid}`
- `source`: `ygocdb`
- `kind`: `faq`
- `fid`
- `primary_card_id`: the card id whose API response supplied the FAQ
- `refer_card_ids`: keys from `faqs[*].refer`
- `title_html`, `question_html`, `answer_html`
- `title_plain`, `question_plain`, `answer_plain`
- `date`
- `snapshot_at` or source snapshot version

For `supplement`, store:

- `ruling_id`: `ygocdb:supplement:{card_id}`
- `source`: `ygocdb`
- `kind`: `supplement`
- `primary_card_id`
- `refer_card_ids`: keys from `supplement.refer`
- `text_html`
- `text_plain`
- `date`

Alternative considered: scrape and parse ygocdb card pages. The API is better because it exposes structured `faqs`, `supplement`, and `refer` data directly.

### Decision: Preserve both HTML and plain text

ygocdb FAQ and supplement fields contain HTML links with card references. Store raw HTML for auditability and plain text for retrieval/display. Extract `data-refer` links from HTML as a secondary signal, but treat the API `refer` object as the authoritative related-card list.

Alternative considered: strip HTML immediately and keep only text. That simplifies retrieval but loses source structure and makes it harder to explain why a ruling applies to a related card.

### Decision: Build direct and reverse ruling indexes

Create indexes independent from card effect documents:

- `faq_by_fid[fid]`
- `evidence_by_id[ruling_id]`
- `faq_ids_by_primary_card_id[card_id]`
- `faq_ids_by_refer_card_id[card_id]`
- sparse and optional dense indexes over title, question, answer, supplement text, card names, and mechanism tags

Retrieval must search both exact involved cards and effect-mechanism queries derived from the scenario. A ruling returned because `primary_card_id` matches should be labeled direct-card evidence. A ruling returned because `refer_card_ids` contains the card should be labeled related-card evidence. A ruling returned through similar-effect or mechanism search should be labeled analogous evidence.

Alternative considered: store rulings as plain text in the existing card index. That would blur card-effect retrieval and ruling retrieval, and it would make it harder to show evidence provenance.

### Decision: Retrieve official rulings and OCG rules in parallel

Official ruling lookup and OCG rules lookup should both run for an adjudication request. Official rulings often answer specific cases; rule documents cover general mechanics when no exact ruling exists or when a ruling needs explanation. They are complementary, not strictly fallback stages.

Alternative considered: only query rules after ruling lookup fails. That would reduce cost but makes explanations weaker and increases the risk of missing a general rule that changes how an analogous ruling should be interpreted.

### Decision: Model analogous ruling retrieval explicitly

The adjudicator should find analogous rulings by:

- Exact involved-card ruling lookup.
- Related-card lookup from `faqs[*].refer`.
- Similar-effect retrieval over local card text to identify cards with equivalent mechanisms.
- Ruling retrieval using normalized mechanism tags such as "destruction replacement", "destroy then special summon", "if destroyed", "instead", "while resolving".

The final answer must label analogous rulings separately from exact-card rulings.

Alternative considered: rely on dense search alone. Dense search is useful recall, but without tags and exact related-card fields it can rank textually similar but legally irrelevant rulings too highly.

### Decision: Parse OCG rules EPUB by table-of-contents sections

Use the local `ocg-rule-readthedocs-io-zh-cn-latest.epub` as the rules source. The EPUB is a zip container with Sphinx/ReadTheDocs XHTML files and table-of-contents metadata in `nav.xhtml` and `toc.ncx`.

Rules ingestion should:

- Read `nav.xhtml` or `toc.ncx` to preserve chapter hierarchy.
- Split documents by XHTML path and anchor when the TOC provides anchors.
- Store stable ids such as `ocgrules:{xhtml_path}#{anchor}`.
- Store chapter path, section title, source path, anchor, raw HTML fragment when practical, and normalized plain text.
- Prefer section-level chunks over fixed-size arbitrary chunks, with secondary chunking only for very long sections.

Relevant first-pass sections for destruction/replacement scenarios include `卡片的抗性 > 代替破坏`, `特定效果的处理方法 > 多步效果的适用`, `特定效果的处理方法 > 卡片从场上离开·被破坏·送去墓地·除外的状况能否发动`, `条件特殊召唤和发动效果特殊召唤`, and `连锁基础 > 如何结算连锁`.

Alternative considered: extract all XHTML text and chunk by character count only. That is easier but loses rule-document hierarchy and produces weaker citations.

### Decision: Use LLM reasoning only over supplied evidence

The LLM can synthesize the final adjudication, but its prompt must include structured evidence ids and require every legal claim to cite evidence ids. If no evidence supports a claim, the output must mark it as inference or uncertainty.

For higher reliability, add a deterministic post-check that rejects answers citing unknown evidence ids or claiming a direct official ruling when only analogous evidence was retrieved.

Alternative considered: let the LLM reason freely with retrieved context. This is simpler but makes hallucinated judge logic hard to detect.

### Decision: Defer `ygoenv` to a simulator adapter

Add a future adapter boundary, for example `SimulationVerifier`, that accepts a normalized scenario plan and returns observations. Initial implementation can return `not_available` or `not_encoded` while document-grounded adjudication works.

`ygoenv` should be implemented last because it requires:

- Mapping named scenario state into engine card ids, zones, controllers, and chain timing.
- Selecting effect activations and replacement choices.
- Detecting whether the engine scripts/data contain the relevant cards and current rulings.
- Comparing simulator observations against document-grounded reasoning without treating the simulator as always authoritative.

Alternative considered: build simulator first. That would delay useful adjudication and is brittle until scenario encoding is reliable.

## Risks / Trade-offs

## Post-Implementation Normalization Decisions

These decisions were discovered and implemented while validating the adjudication workflow, after the initial task list was already marked complete.

### Decision: Decode card effect tags from `strings.conf`

Card effect tags are decoded from the YGOPro `datas.category` bitmask and the localized labels in `strings.conf`, specifically the `!system 1100+` entries from `https://cdn02.moecube.com:444/ygopro-database/zh-CN/strings.conf`.

The implementation keeps the raw `category` integer for auditability, decodes stable internal bit keys, reads localized labels from a local `strings.conf` when available, and falls back to an internal map that follows the same `!system 1100+` bit order. Decoded tags are added to card document text and metadata so both card retrieval and adjudication retrieval can use them.

Example: `完美电子多元驱动蛇·神龙` has category `0x3400002`, which decodes to `怪兽破坏`, `LP伤害`, `破坏耐性`, and `效果耐性`.

### Decision: Use effect tags to improve analogous evidence retrieval

Resolved cards contribute decoded effect category labels to the mechanism query used for FAQ and OCG rules retrieval. FAQ search may also boost rulings whose primary or related card ids have overlapping decoded effect tags. This improves recall for effects that are legally similar but do not share enough surface wording.

FAQ tagging itself remains a future LLM-assisted possibility. The current implementation tags cards, not FAQ entries, and uses card-to-ruling relationships to transfer tag signals.

### Decision: Parse OCG rules EPUB by anchor-local sections

OCG rules EPUB parsing must not assign a whole XHTML page to every TOC anchor. The parser should locate the anchor in the XHTML DOM, extract only that heading or section's local content, stop before the next peer heading, and then chunk long sections by paragraph-aware grouping.

The rules index should skip page-level `#top` entries when concrete anchors exist for the same XHTML file, and sparse rules search should weight section title and chapter path higher than body text.

### Decision: Require structured final adjudication JSON

The final LLM answer prompt is a separately testable prompt builder. It requires strict JSON with fixed fields: `conclusion`, `answers`, `evidence_use`, and `uncertainties`.

The prompt requires each parsed user question to be answered separately. If a question contains a hypothetical branch, the final answer must include both the actual-case conclusion and the hypothetical-branch conclusion instead of stopping at "the premise is false." A deterministic validator rejects invalid JSON and unknown evidence ids.

### Decision: Keep local data downloads proxy-explicit

Bulk downloads for ygocdb ruling JSON and `strings.conf` default to no system proxy. Proxy usage should be explicit opt-in. This keeps cache refresh behavior predictable in environments where a system proxy such as v2ray is configured but should not be used for these data snapshots.

- [Risk] Official ruling data may be incomplete or stale -> Mitigation: include source snapshot metadata, warnings, and confidence levels.
- [Risk] Querying ygocdb live for every request may be slow or brittle -> Mitigation: cache API responses and build local indexes from snapshots.
- [Risk] `faqs[*].refer` can include many cards that are related by examples rather than exact scenario identity -> Mitigation: label refer matches as related-card evidence, not direct-card evidence.
- [Risk] Similar-effect retrieval may surface legally different effects -> Mitigation: separate exact, related, and analogous evidence and require the answer to label them.
- [Risk] LLM scenario parsing may choose the wrong card or effect -> Mitigation: card resolver ambiguity reporting and structured extraction tests with representative scenarios.
- [Risk] OCG rules EPUB parsing may produce noisy passages -> Mitigation: parse into sections with stable headings and cite section ids, not arbitrary chunks only.
- [Risk] `ygoenv` behavior may differ from official/current rulings -> Mitigation: present simulation as verification evidence, not as the sole authority.
- [Risk] Query latency will increase -> Mitigation: cache parsed rule/ruling indexes and keep simulation disabled unless explicitly requested.

## Migration Plan

1. Add ygocdb API cache ingestion and EPUB section ingestion without changing current query behavior.
2. Add the adjudication service and tests for scenario parsing, card resolution, evidence retrieval, and final structured output.
3. Expose CLI and Web entry points for adjudication.
4. Add optional LLM synthesis constrained to evidence.
5. Add `ygoenv` simulation adapter in a later change once document-grounded adjudication is stable.

## Open Questions

- What cache refresh policy should be used for ygocdb API responses?
- Should adjudication require LLM credentials, or should there be a reduced deterministic mode that returns evidence without final reasoning?
- What confidence threshold should cause the system to refuse a direct answer and ask for clarification?
- How should card aliases and translated names be normalized across YGOPro card text, ruling sources, and user input?
