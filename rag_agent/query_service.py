from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_agent.agent import (
    CardRagAgent,
    RetrievedCard,
    build_answer_prompt,
    format_retrieval_answer,
)
from rag_agent.cards import Card
from rag_agent.config import Settings
from rag_agent.db import load_cards
from rag_agent.llm import create_deepseek_chat_model
from rag_agent.retrieval.embeddings import BgeM3Embeddings
from rag_agent.retrieval.reranker import BgeReranker, LlmReranker, SubprocessReranker
from rag_agent.retrieval.vector import ChromaConfig, ChromaVectorStore


@dataclass(frozen=True)
class QueryRequest:
    query: str
    db_path: Path
    top_k: int = 10
    semantic: bool = False
    rerank: bool = False
    rerank_provider: str | None = None
    use_llm: bool = False
    rerank_candidates: int = 20


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    results: list[dict[str, Any]]
    warnings: list[str]
    structured_query: dict[str, Any] | None = None
    filter_diagnostics: dict[str, Any] | None = None


def execute_query(request: QueryRequest, settings: Settings) -> QueryResponse:
    cards = load_cards(request.db_path)
    warnings: list[str] = []

    agent = build_agent(
        cards,
        settings,
        semantic=request.semantic,
        rerank=request.rerank,
        use_llm=request.use_llm,
        rerank_provider=request.rerank_provider,
        warnings=warnings,
    )
    retrieved = agent.retrieve(
        request.query,
        top_k=request.top_k,
        rerank_candidates=request.rerank_candidates,
    )
    answer = answer_from_retrieved_cards(
        agent,
        request.query,
        retrieved,
        use_llm=request.use_llm,
    )
    return QueryResponse(
        answer=answer,
        results=[retrieved_card_to_dict(card) for card in retrieved],
        warnings=warnings,
        structured_query=agent.last_structured_query.to_dict(),
        filter_diagnostics=agent.last_filter_diagnostics.to_dict(),
    )


def build_agent(
    cards: list[Card],
    settings: Settings,
    *,
    semantic: bool,
    rerank: bool,
    use_llm: bool,
    rerank_provider: str | None = None,
    warnings: list[str] | None = None,
) -> CardRagAgent:
    dense_retriever = None
    if semantic:
        embeddings = BgeM3Embeddings(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        candidate_dense_retriever = ChromaVectorStore(
            ChromaConfig(settings.chroma_persist_dir),
            embeddings,
        )
        indexed_count = candidate_dense_retriever.count()
        if indexed_count < len(cards):
            message = (
                f"Chroma index has {indexed_count} documents but cards DB has {len(cards)}; "
                "skipping dense retrieval until the full index is built."
            )
            if warnings is not None:
                warnings.append(message)
        else:
            dense_retriever = candidate_dense_retriever

    provider = resolve_rerank_provider_from_values(
        rerank=rerank,
        rerank_provider=rerank_provider,
        settings=settings,
    )

    llm = None
    reranker = None
    if provider == "llm":
        llm_for_rerank = create_deepseek_chat_model(settings)
        reranker = LlmReranker(
            llm=llm_for_rerank,
            max_candidates=settings.llm_rerank_max_candidates,
            warning_callback=warnings.append if warnings is not None else None,
        )
    elif provider == "local":
        if settings.embedding_device.startswith("cuda"):
            reranker_device = (
                settings.reranker_device
                if settings.reranker_device_explicit
                else "auto"
            )
            reranker = SubprocessReranker(
                settings.reranker_model,
                device=reranker_device,
            )
        else:
            reranker = BgeReranker(
                settings.reranker_model,
                device=settings.reranker_device,
            )

    if use_llm:
        llm = create_deepseek_chat_model(settings)
    return CardRagAgent(
        cards,
        dense_retriever=dense_retriever,
        reranker=reranker,
        llm=llm,
        warning_callback=warnings.append if warnings is not None else None,
    )


def resolve_rerank_provider(request: QueryRequest, settings: Settings) -> str:
    return resolve_rerank_provider_from_values(
        rerank=request.rerank,
        rerank_provider=request.rerank_provider,
        settings=settings,
    )


def resolve_rerank_provider_from_values(
    *,
    rerank: bool,
    rerank_provider: str | None,
    settings: Settings,
) -> str:
    if rerank_provider:
        provider = rerank_provider
    elif rerank:
        provider = "local"
    else:
        provider = settings.rerank_provider

    normalized = provider.strip().lower()
    aliases = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "bge": "local",
        "local-bge": "local",
        "llm-rerank": "llm",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "local", "llm"}:
        raise ValueError(
            "rerank provider must be one of: none, local, llm."
        )
    return normalized


def answer_from_retrieved_cards(
    agent: CardRagAgent,
    query: str,
    cards: list[RetrievedCard],
    *,
    use_llm: bool,
) -> str:
    if not use_llm:
        return format_retrieval_answer(query, cards)

    if agent.llm is None:
        raise RuntimeError("LLM is not configured.")
    response = agent.llm.invoke(build_answer_prompt(query, cards))
    content = getattr(response, "content", response)
    return str(content)


def retrieved_card_to_dict(card: RetrievedCard) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "name": card.name,
        "score": card.score,
        "source_text": card.source_text,
        "reason": card.reason,
    }


def build_structured_response(
    response: QueryResponse,
    *,
    max_block_chars: int | None = None,
) -> dict[str, Any]:
    blocks = [
        build_card_block(index, result, max_block_chars=max_block_chars)
        for index, result in enumerate(response.results, start=1)
    ]
    return {
        "version": 1,
        "summary": {
            "result_count": len(response.results),
            "warning_count": len(response.warnings),
            "warnings": response.warnings,
        },
        "structured_query": response.structured_query,
        "filter_diagnostics": response.filter_diagnostics,
        "blocks": blocks,
    }


def build_card_block(
    index: int,
    result: dict[str, Any],
    *,
    max_block_chars: int | None = None,
) -> dict[str, Any]:
    card_id = int(result["card_id"])
    name = str(result.get("name", ""))
    score = float(result.get("score", 0.0))
    source_text = str(result.get("source_text", ""))
    reason = str(result.get("reason", ""))
    text = "\n".join(
        [
            f"{index}. {name}",
            f"ID: {card_id}",
            f"Score: {score:.4f}",
            f"效果: {source_text}",
            f"理由: {reason}",
        ]
    )
    truncated_text = truncate_text(text, max_block_chars)
    return {
        "type": "card",
        "index": index,
        "card_id": card_id,
        "name": name,
        "score": score,
        "text": truncated_text,
        "truncated": truncated_text != text,
        "fields": {
            "card_id": card_id,
            "name": name,
            "score": score,
            "source_text": source_text,
            "reason": reason,
        },
    }


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[: max_chars - 1].rstrip() + "…"
