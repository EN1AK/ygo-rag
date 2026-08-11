from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rag_agent.adjudication import (
    AdjudicationProgressCallback,
    AdjudicationResponse,
    RuleIndex,
    RulingAdjudicator,
    YgocdbCache,
    YgocdbRulingIndex,
    parse_scenario,
    resolve_card_mentions,
)
from rag_agent.config import Settings
from rag_agent.db import load_cards
from rag_agent.llm import create_deepseek_chat_model
from rag_agent.retrieval.embeddings import (
    BgeM3Embeddings,
    SubprocessBgeM3Embeddings,
)
from rag_agent.ruling_index import (
    PersistentRulingRetriever,
    RulingVectorStore,
    load_rule_documents_from_epub,
    load_ygocdb_ruling_documents_from_cache,
    validate_manifest,
)


@dataclass(frozen=True)
class AdjudicationRequest:
    scenario: str
    db_path: Path
    ygocdb_cache_path: Path | None = None
    rules_epub_path: Path | None = None
    use_llm: bool = False
    simulate: bool = False
    allow_ygocdb_fetch: bool = False
    use_agent: bool = False
    include_trace: bool = False
    progress_callback: AdjudicationProgressCallback | None = None


def execute_adjudication(
    request: AdjudicationRequest,
    settings: Settings,
) -> AdjudicationResponse:
    cards = load_cards(request.db_path)
    ruling_card_ids = _ruling_card_ids_for_request(request, cards, settings)
    ruling_index = build_ruling_index(request, settings, ruling_card_ids)
    rule_index = build_rule_index(request, settings)
    llm = create_deepseek_chat_model(settings) if request.use_llm else None
    persistent_retrievers = (
        build_persistent_ruling_retrievers(request, settings, cards, llm=llm)
        if request.use_agent
        else ()
    )
    if request.use_agent:
        from rag_agent.adjudication_agent import AdjudicationAgent

        response = AdjudicationAgent(
            cards,
            settings=settings,
            ruling_index=ruling_index,
            rule_index=rule_index,
            ruling_retrievers=persistent_retrievers,
            llm=llm,
            progress_callback=request.progress_callback,
        ).run(request.scenario)
        if not (request.include_trace or settings.adjudication_trace_default):
            response = replace(response, agent_trace=())
        return response

    adjudicator = RulingAdjudicator(
        cards,
        ruling_index=ruling_index,
        rule_index=rule_index,
        llm=llm,
    )
    return adjudicator.adjudicate(request.scenario, simulate=request.simulate)


def _ruling_card_ids_for_request(
    request: AdjudicationRequest,
    cards,
    settings: Settings,
) -> list[int]:
    if request.use_agent:
        from rag_agent.adjudication_agent import (
            candidate_mention_queries,
            search_card_candidates,
            select_card_resolution,
        )
        from rag_agent.retrieval.sparse import SparseCardRetriever

        scenario = parse_scenario(request.scenario, known_cards=cards)
        queries = candidate_mention_queries(request.scenario, scenario)
        retriever = SparseCardRetriever.from_cards(cards)
        card_ids: list[int] = []
        for query in queries:
            resolution = select_card_resolution(
                query,
                search_card_candidates(query, cards, top_k=5, retriever=retriever),
                confidence_threshold=settings.adjudication_card_confidence_threshold,
                ambiguity_margin=settings.adjudication_card_ambiguity_margin,
            )
            if resolution.status == "resolved" and resolution.selected_card_id is not None:
                card_ids.append(resolution.selected_card_id)
            elif resolution.status == "ambiguous":
                card_ids.extend(candidate.card_id for candidate in resolution.candidates[:3])
        return list(dict.fromkeys(card_ids))
    if not request.allow_ygocdb_fetch:
        return [card.card_id for card in cards]
    scenario = parse_scenario(request.scenario, known_cards=cards)
    resolved = resolve_card_mentions(scenario.card_mentions, cards)
    return [
        item.matches[0].card_id
        for item in resolved
        if len(item.matches) == 1
    ]


def build_ruling_index(
    request: AdjudicationRequest,
    settings: Settings,
    card_ids: list[int],
) -> YgocdbRulingIndex | None:
    cache_path = request.ygocdb_cache_path or settings.ygocdb_cache_path
    if not cache_path.exists() and not request.allow_ygocdb_fetch:
        return None
    cache = YgocdbCache(cache_path)
    payloads: list[dict[str, Any]] = []
    for card_id in card_ids:
        payload = cache.get_card_payload(
            card_id,
            allow_fetch=request.allow_ygocdb_fetch,
            timeout=settings.deepseek_timeout_seconds,
            use_proxy=False,
        )
        if payload is not None:
            payloads.append(payload)
    if not payloads:
        return None
    return YgocdbRulingIndex.from_payloads(payloads)


def build_rule_index(
    request: AdjudicationRequest,
    settings: Settings,
) -> RuleIndex | None:
    rules_path = request.rules_epub_path or settings.ocg_rules_epub_path
    if not rules_path.exists():
        return None
    return RuleIndex.from_epub(rules_path)


def build_persistent_ruling_retrievers(
    request: AdjudicationRequest,
    settings: Settings,
    cards,
    *,
    llm: Any | None = None,
) -> tuple[PersistentRulingRetriever, ...]:
    embeddings = (
        SubprocessBgeM3Embeddings(
            settings.embedding_model,
            device=settings.embedding_device,
            show_progress=False,
        )
        if settings.embedding_device.startswith("cuda")
        else BgeM3Embeddings(
            settings.embedding_model,
            device=settings.embedding_device,
            show_progress=False,
        )
    )
    retrievers: list[PersistentRulingRetriever] = []
    faq_documents, _faq_sources = load_ygocdb_ruling_documents_from_cache(
        request.ygocdb_cache_path or settings.ygocdb_cache_path,
        cards,
    )
    faq_status = validate_manifest(
        settings.ruling_faq_manifest_path,
        index_kind="faq",
        collection_name=settings.ruling_faq_collection_name,
        embedding_model=settings.embedding_model,
    )
    if faq_documents and faq_status.usable:
        retrievers.append(
            PersistentRulingRetriever(
                faq_documents,
                dense_store=RulingVectorStore(
                    settings.ruling_faq_index_path,
                    settings.ruling_faq_collection_name,
                    embeddings,
                ),
                manifest_status=faq_status,
                llm=llm,
                llm_max_candidates=settings.llm_rerank_max_candidates,
            )
        )
    rule_documents, _rule_sources = load_rule_documents_from_epub(
        request.rules_epub_path or settings.ocg_rules_epub_path,
    )
    rules_status = validate_manifest(
        settings.ruling_rules_manifest_path,
        index_kind="rules",
        collection_name=settings.ruling_rules_collection_name,
        embedding_model=settings.embedding_model,
    )
    if rule_documents and rules_status.usable:
        retrievers.append(
            PersistentRulingRetriever(
                rule_documents,
                dense_store=RulingVectorStore(
                    settings.ruling_rules_index_path,
                    settings.ruling_rules_collection_name,
                    embeddings,
                ),
                manifest_status=rules_status,
                llm=llm,
                llm_max_candidates=settings.llm_rerank_max_candidates,
            )
        )
    return tuple(retrievers)


def adjudication_response_to_dict(response: AdjudicationResponse) -> dict[str, Any]:
    return response.to_dict()


def render_adjudication_answer(answer: str) -> str:
    """Render a structured adjudication answer as concise natural Chinese."""
    try:
        payload = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return str(answer).strip()
    if not isinstance(payload, dict):
        return str(answer).strip()

    lines: list[str] = []
    conclusion = str(payload.get("conclusion") or "").strip()
    if conclusion:
        lines.append(f"裁定：{conclusion}")
    answers = payload.get("answers")
    if isinstance(answers, list):
        for index, item in enumerate(answers, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            case_conclusion = str(item.get("case_conclusion") or "").strip()
            hypothetical = str(item.get("hypothetical_conclusion") or "").strip()
            prefix = f"问题 {index}"
            if question:
                lines.append(f"{prefix}：{question}")
            if case_conclusion:
                lines.append(f"结论：{case_conclusion}")
            if hypothetical:
                lines.append(f"假设分支：{hypothetical}")

    source_status = payload.get("source_status")
    if isinstance(source_status, dict):
        labels = {
            "found": "已找到",
            "not_found": "未找到",
            "not_searched": "未检索",
        }
        faq = labels.get(str(source_status.get("faq")), "未知")
        rules = labels.get(str(source_status.get("rules")), "未知")
        lines.append(f"来源状态：FAQ {faq}；规则文档 {rules}。")
        failure_reason = str(source_status.get("failure_reason") or "").strip()
        if failure_reason:
            lines.append(f"证据说明：{failure_reason}")

    uncertainties = payload.get("uncertainties")
    if isinstance(uncertainties, list):
        values = [str(item).strip() for item in uncertainties if str(item).strip()]
        if values:
            lines.append("不确定项：" + "；".join(values))
    return "\n".join(lines) if lines else str(answer).strip()
