from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from rag_agent.cards import Card
from rag_agent.retrieval.hybrid import Candidate, reciprocal_rank_fusion
from rag_agent.retrieval.sparse import SparseCardRetriever
from rag_agent.retrieval.vector import structured_filters_to_chroma_where
from rag_agent.structured_query import (
    FilterDiagnostics,
    StructuredQuery,
    card_matches_filters,
    parse_structured_query,
)


@dataclass(frozen=True)
class RetrievedCard:
    card_id: int
    name: str
    score: float
    source_text: str
    reason: str


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        ...


class Llm(Protocol):
    def invoke(self, prompt: str):
        ...


def format_retrieval_answer(query: str, cards: Sequence[RetrievedCard]) -> str:
    lines = [f"查询：{query}", "", "相近效果候选："]
    if not cards:
        lines.append("未找到足够相近的候选卡。")
        return "\n".join(lines)

    for index, card in enumerate(cards, start=1):
        lines.extend(
            [
                f"{index}. {card.name} (ID: {card.card_id}, score: {card.score:.4f})",
                f"   原文：{card.source_text}",
                f"   相似原因：{card.reason}",
            ]
        )
    return "\n".join(lines)


def build_answer_prompt(query: str, cards: Sequence[RetrievedCard]) -> str:
    context_lines = []
    for index, card in enumerate(cards, start=1):
        context_lines.append(
            "\n".join(
                [
                    f"[{index}] 卡名：{card.name}",
                    f"ID：{card.card_id}",
                    f"检索分数：{card.score:.4f}",
                    f"原文：{card.source_text}",
                    f"检索说明：{card.reason}",
                ]
            )
        )
    context = "\n\n".join(context_lines) if context_lines else "无候选卡。"
    return (
        "你是一个游戏王卡片效果检索助手。只根据给定候选卡原文回答，"
        "不要编造未提供的卡片或效果。\n\n"
        f"用户问题：{query}\n\n"
        f"候选卡：\n{context}\n\n"
        "请用中文输出：1) 最相近的卡片列表；2) 每张为什么相近；"
        "3) 如果只是关键词相似但效果方向不同，要明确说明。"
    )


def _candidate_to_retrieved_card(candidate: Candidate) -> RetrievedCard:
    metadata = candidate.metadata
    return RetrievedCard(
        card_id=candidate.card_id,
        name=str(metadata.get("name", "")),
        score=candidate.score,
        source_text=str(metadata.get("desc") or metadata.get("text") or ""),
        reason=_reason_for_candidate(candidate),
    )


def _reason_for_candidate(candidate: Candidate) -> str:
    if candidate.source == "llm_reranker":
        reason = str(candidate.metadata.get("llm_judge_reason") or "").strip()
        if reason:
            return f"LLM judge reranked this candidate: {reason}"
    return _reason_for_source(candidate.source)


def _reason_for_source(source: str) -> str:
    if source == "llm_reranker":
        return "Candidates reranked by LLM judge."
    if source == "reranker":
        return "Hybrid candidates reranked by local bge-reranker-v2-m3."
    if source == "hybrid":
        return "Hybrid search matched dense semantic and/or sparse lexical signals."
    if source == "dense":
        return "Dense vector search matched semantic effect text."
    if source == "sparse":
        return "Sparse baseline matched card name/effect terms."
    return f"Retrieved by {source}."


class CardRagAgent:
    def __init__(
        self,
        cards: Sequence[Card],
        *,
        dense_retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        llm: Llm | None = None,
        warning_callback=None,
    ) -> None:
        self.cards = list(cards)
        self.sparse_retriever = SparseCardRetriever.from_cards(cards)
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.llm = llm
        self.warning_callback = warning_callback
        self.last_structured_query = parse_structured_query("")
        self.last_filter_diagnostics = FilterDiagnostics(total_candidates=len(self.cards))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        rerank_candidates: int = 20,
    ) -> list[RetrievedCard]:
        structured_query = parse_structured_query(query)
        self.last_structured_query = structured_query
        candidate_cards = self._filter_cards(structured_query)
        diagnostics_warnings: list[str] = []

        if structured_query.has_filters and not candidate_cards:
            warning = "Structured query filters matched no cards; returning no retrieval results."
            diagnostics_warnings.append(warning)
            self._warn(warning)

        self.last_filter_diagnostics = FilterDiagnostics(
            applied=structured_query.has_filters,
            total_candidates=len(self.cards),
            filtered_candidates=len(candidate_cards),
            warnings=tuple(diagnostics_warnings),
        )
        if structured_query.has_filters and not candidate_cards:
            return []

        retrieval_query = self._expand_query_with_referenced_card(
            structured_query.effect_query or query
        )
        recall_top_k = (
            min(len(candidate_cards), max(top_k, 100))
            if structured_query.has_filters
            else top_k
        )
        sparse_retriever = (
            SparseCardRetriever.from_cards(candidate_cards)
            if structured_query.has_filters
            else self.sparse_retriever
        )
        sparse_candidates = sparse_retriever.search(retrieval_query, top_k=recall_top_k)
        ranked_lists: list[list[Candidate]] = [sparse_candidates]

        if self.dense_retriever is not None:
            dense_top_k = recall_top_k
            dense_filter = (
                structured_filters_to_chroma_where(structured_query.filters)
                if structured_query.has_filters
                else None
            )
            dense_candidates = self._search_dense(
                retrieval_query,
                top_k=dense_top_k,
                metadata_filter=dense_filter,
                fallback_top_k=max(top_k, top_k * 20),
            )
            if structured_query.has_filters:
                allowed_ids = {card.card_id for card in candidate_cards}
                dense_candidates = [
                    candidate
                    for candidate in dense_candidates
                    if candidate.card_id in allowed_ids
                ][:top_k]
            ranked_lists.append(dense_candidates)

        candidate_limit = max(
            top_k,
            rerank_candidates if self.reranker is not None else top_k * 3,
            recall_top_k if structured_query.has_filters else top_k,
        )
        candidates = reciprocal_rank_fusion(ranked_lists, top_k=candidate_limit)
        if self.reranker is not None:
            candidates = self.reranker.rerank(retrieval_query, candidates)

        return [_candidate_to_retrieved_card(candidate) for candidate in candidates[:top_k]]

    def query(
        self,
        query: str,
        *,
        top_k: int = 10,
        rerank_candidates: int = 20,
    ) -> str:
        cards = self.retrieve(
            query,
            top_k=top_k,
            rerank_candidates=rerank_candidates,
        )
        if self.llm is None:
            return format_retrieval_answer(query, cards)

        prompt = build_answer_prompt(query, cards)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content)

    def _expand_query_with_referenced_card(self, query: str) -> str:
        referenced = [
            card
            for card in self.cards
            if card.name and len(card.name) >= 2 and card.name in query
        ]
        if not referenced:
            return query
        referenced.sort(key=lambda card: len(card.name), reverse=True)
        card = referenced[0]
        return f"{query}\n参考卡：{card.name}\n参考卡效果：{card.description}"

    def _filter_cards(self, structured_query: StructuredQuery) -> list[Card]:
        if not structured_query.has_filters:
            return self.cards
        return [
            card
            for card in self.cards
            if card_matches_filters(card, structured_query.filters)
        ]

    def _warn(self, warning: str) -> None:
        if self.warning_callback is not None:
            self.warning_callback(warning)

    def _search_dense(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
        fallback_top_k: int,
    ) -> list[Candidate]:
        if self.dense_retriever is None:
            return []
        if metadata_filter is None:
            return self.dense_retriever.search(query, top_k=top_k)
        try:
            return self.dense_retriever.search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            self._warn(
                "Dense metadata filtering failed; falling back to dense post-filter "
                f"candidate order. Rebuild Chroma with `build-index --reset` to enable "
                f"metadata filtering. Cause: {exc}"
            )
            return self.dense_retriever.search(query, top_k=fallback_top_k)
