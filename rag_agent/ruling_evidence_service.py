"""Public pure-RAG adjudication backed by the bounded evidence loop."""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from time import perf_counter
from unicodedata import normalize

from rag_agent.adjudication import (
    AdjudicationProgressCallback, AdjudicationProgressEvent, AdjudicationResponse,
    EvidenceItem, Llm, ReasoningStep, ResolvedCard,
    RuleIndex, build_card_effect_evidence,
)
from rag_agent.application.adjudication.evidence_loop import run_evidence_loop
from rag_agent.card_alias_registry import DEFAULT_VERIFIED_CARD_ALIASES
from rag_agent.cards import Card
from rag_agent.ruling_index import PersistentRulingRetriever


def _card_name_key(name: str) -> str:
    return "".join(normalize("NFKC", name).casefold().split())


def resolve_evidence_card(
    query: str,
    cards: Sequence[Card],
    verified_resolver: Callable[[str], ResolvedCard] | None = None,
    is_verified_alias: Callable[[str, int], bool] | None = None,
) -> ResolvedCard:
    key = _card_name_key(query)
    default_id = next((
        alias[0] for name, alias in DEFAULT_VERIFIED_CARD_ALIASES.items()
        if _card_name_key(name) == key
    ), None)
    if verified_resolver is None or key.isdecimal():
        candidates = cards
    else:
        candidates = verified_resolver(query).matches
    matches = tuple(
        card for card in candidates
        if key == str(card.card_id) or key == _card_name_key(card.name)
        or card.card_id == default_id
        or (is_verified_alias is not None and is_verified_alias(query, card.card_id))
    )
    return ResolvedCard(query, matches, ambiguous=len(matches) > 1)


def answer_with_evidence_loop(
    base: AdjudicationResponse,
    llm: Llm,
    *,
    rule_index: RuleIndex | None,
    retrievers: Sequence[PersistentRulingRetriever],
    resolve: Callable[[str], ResolvedCard],
    question_source: str | None = None,
    include_trace: bool = False,
    progress_callback: AdjudicationProgressCallback | None = None,
) -> AdjudicationResponse:
    known = {item.evidence_id: item for item in base.evidence}
    resolved = list(base.resolved_cards)
    stages = []
    retrieval_warnings: set[str] = set()

    aliases_by_evidence: dict[str, set[str]] = {}

    # Persist verified name-to-id bindings alongside the card facts.
    def lookup_card(query: str) -> list[EvidenceItem]:
        card = resolve(query)
        items = build_card_effect_evidence((card,))
        existing_ids = {match.card_id for item in resolved for match in item.matches}
        if not items:
            return []
        if card.matches[0].card_id not in existing_ids:
            resolved.append(card)
        item = items[0]
        if _card_name_key(query) != _card_name_key(card.matches[0].name):
            aliases = aliases_by_evidence.setdefault(item.evidence_id, set())
            aliases.add(query)
            item = replace(
                item,
                text="已核验卡名对应：" + "、".join(sorted(aliases)) + " → " + item.title + "\n" + item.text,
                metadata={**item.metadata, "verified_mentions": sorted(aliases)},
            )
        known[item.evidence_id] = item
        return [item]

    scenario_key = _card_name_key(base.scenario.original_text)
    for alias in DEFAULT_VERIFIED_CARD_ALIASES:
        if _card_name_key(alias) in scenario_key:
            lookup_card(alias)
    initial = tuple(known.values())

    card_ids = tuple(dict.fromkeys(
        item.matches[0].card_id for item in resolved if len(item.matches) == 1
    ))

    def retrieve(queries: dict[str, str]) -> dict[str, list[EvidenceItem]]:
        pools = {}
        # Separate source budgets preserve both dense and lexical candidates.
        for key, query in queries.items():
            limit = 3 if key == "original" else 5
            items = []
            for retriever in retrievers:
                hits = retriever.search(query, card_ids=card_ids, final_top_k=limit)
                items.extend(candidate.to_evidence() for candidate in hits.candidates)
                retrieval_warnings.update(hits.warnings)
            if rule_index is not None:
                items.extend(section.to_evidence() for section in rule_index.search(query, top_k=limit))
            for item in items:
                known.setdefault(item.evidence_id, item)
            pools[key] = [known[item.evidence_id] for item in items]
        return pools

    def invoke(stage: str, prompt: str) -> str:
        if progress_callback is not None:
            messages = {
                "plan": "正在拆分裁定问题并制定检索计划。",
                "coverage": "正在检查卡文、裁定与规则依据。",
                "coverage_retry": "正在复核补查到的证据。",
                "answer": "正在生成带引用的裁定。",
            }
            progress_callback(AdjudicationProgressEvent(stage=stage, message=messages[stage]))
        start = perf_counter()
        try:
            reply = llm.invoke(prompt)
        except Exception as exc:
            # Provider exception details can include transport parameters or request data.
            raise ValueError(f"Model call failed at {stage} ({type(exc).__name__}).") from exc
        content = getattr(reply, "content", reply)
        if not isinstance(content, str):
            raise ValueError(f"{stage} returned non-text content.")
        stage_info = {"stage": stage, "elapsed_ms": round((perf_counter() - start) * 1000)}
        usage = getattr(reply, "usage_metadata", None)
        if isinstance(usage, dict):
            stage_info["usage"] = {
                key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")
                if isinstance(usage.get(key), int)
            }
        stages.append(stage_info)
        return content

    # A follow-up may include earlier turns for context but asks only about the current turn.
    questions = (question_source,) if question_source else base.scenario.questions
    scenario = base.scenario.original_text
    if question_source and question_source.strip() != scenario.strip():
        scenario = f"已知对话背景：\n{scenario}\n本轮用户问题（仅回答本轮）：\n{question_source}"

    try:
        result = run_evidence_loop(scenario, questions, initial, invoke, retrieve, lookup_card)
    except ValueError:
        warning = "证据循环未能完成，请重试裁定。"
        return replace(
            base,
            answer=json.dumps({"conclusion": warning, "answers": []}, ensure_ascii=False),
            reasoning=(),
            confidence="low",
            warnings=(*base.warnings, warning),
            agent_coverage={"route": "bounded_evidence_loop", "status": "failed"},
            agent_trace=tuple(stages) if include_trace else (),
        )

    packed_ids = dict.fromkeys([
        *(item["evidence_id"] for item in result["packed_evidence"].values()),
        *(item.evidence_id for item in known.values() if item.kind == "card_effect"),
    ])
    packed = tuple(known[evidence_id] for evidence_id in packed_ids)
    checks = result["answers"]
    coverage = result["coverage"]
    clarifications = list(dict.fromkeys(
        f"请补充确认：{item['detail']}" for item in coverage.values() if item["status"] == "missing_fact"
    ))
    reasons = tuple(ReasoningStep(item["conclusion"], tuple(item["evidence_ids"])) for item in checks)
    answers = [
        {
            "question": item["question"],
            "case_conclusion": item["conclusion"] if item["status"] != "missing_fact" else "",
            "hypothetical_conclusion": item["conclusion"] if item["status"] == "missing_fact" else "",
            "reasoning": [{"judgment": item["conclusion"], "evidence_ids": item["evidence_ids"]}],
        }
        for item in checks
    ]
    complete = all(item["status"] == "supported" and item["evidence_ids"] for item in checks)
    answer = {
        "conclusion": " ".join(item["case_conclusion"] for item in answers if item["case_conclusion"])
        or " ".join(clarifications),
        "source_status": "supported" if complete else "partial",
        "answers": answers,
        "uncertainties": clarifications,
    }
    info = {
        "route": "bounded_evidence_loop",
        "status": "complete" if complete else "partial",
        "awaiting_clarification": bool(clarifications),
        "clarification_questions": clarifications,
    }
    if include_trace:
        info["checkpoints"] = result["checkpoints"]
        info["coverage"] = {
            key: {**value, "refs": [
                result["packed_evidence"][ref]["evidence_id"] for ref in value["refs"]
            ]} for key, value in coverage.items()
        }
    return replace(
        base,
        resolved_cards=tuple(resolved),
        evidence=packed,
        reasoning=reasons,
        answer=json.dumps(answer, ensure_ascii=False),
        confidence="medium" if complete else "low",
        warnings=tuple(dict.fromkeys((*base.warnings, *sorted(retrieval_warnings)))),
        agent_coverage=info,
        agent_trace=tuple(stages) if include_trace else (),
    )
