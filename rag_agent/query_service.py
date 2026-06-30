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
from rag_agent.retrieval.reranker import BgeReranker, SubprocessReranker
from rag_agent.retrieval.vector import ChromaConfig, ChromaVectorStore


@dataclass(frozen=True)
class QueryRequest:
    query: str
    db_path: Path
    top_k: int = 10
    semantic: bool = False
    rerank: bool = False
    use_llm: bool = False
    rerank_candidates: int = 20


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    results: list[dict[str, Any]]
    warnings: list[str]


def execute_query(request: QueryRequest, settings: Settings) -> QueryResponse:
    cards = load_cards(request.db_path)
    warnings: list[str] = []

    agent = build_agent(
        cards,
        settings,
        semantic=request.semantic,
        rerank=request.rerank,
        use_llm=request.use_llm,
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
    )


def build_agent(
    cards: list[Card],
    settings: Settings,
    *,
    semantic: bool,
    rerank: bool,
    use_llm: bool,
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

    reranker = None
    if rerank:
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

    llm = create_deepseek_chat_model(settings) if use_llm else None
    return CardRagAgent(
        cards,
        dense_retriever=dense_retriever,
        reranker=reranker,
        llm=llm,
    )


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
