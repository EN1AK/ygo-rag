from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from rag_agent.cards import Card, card_to_document
from rag_agent.retrieval.hybrid import Candidate


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize_text(text: str) -> list[str]:
    normalized = text.lower()
    cjk_chars = _CJK_RE.findall(normalized)
    bigrams = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
    ascii_terms = re.findall(r"[a-z0-9_]+", normalized)
    return cjk_chars + bigrams + ascii_terms


@dataclass(frozen=True)
class SparseDocument:
    card_id: int
    text: str
    metadata: dict
    term_counts: Counter[str]


class SparseCardRetriever:
    def __init__(self, documents: list[SparseDocument]) -> None:
        self.documents = documents
        self.doc_freq: Counter[str] = Counter()
        for document in documents:
            self.doc_freq.update(document.term_counts.keys())

    @classmethod
    def from_cards(cls, cards: Iterable[Card]) -> "SparseCardRetriever":
        documents = []
        for card in cards:
            retrieval_document = card_to_document(card)
            text = retrieval_document.page_content
            documents.append(
                SparseDocument(
                    card_id=card.card_id,
                    text=text,
                    metadata=retrieval_document.metadata,
                    term_counts=Counter(tokenize_text(text)),
                )
            )
        return cls(documents)

    def search(self, query: str, *, top_k: int = 10) -> list[Candidate]:
        query_terms = Counter(tokenize_text(query))
        if not query_terms:
            return []

        scored: list[Candidate] = []
        total_docs = max(len(self.documents), 1)
        for document in self.documents:
            score = 0.0
            for term, query_count in query_terms.items():
                term_count = document.term_counts.get(term, 0)
                if term_count == 0:
                    continue
                idf = math.log((1 + total_docs) / (1 + self.doc_freq[term])) + 1
                score += query_count * term_count * idf

            name = str(document.metadata.get("name", ""))
            if name and name in query:
                score += 1000.0

            if score > 0:
                scored.append(
                    Candidate(
                        card_id=document.card_id,
                        score=score,
                        source="sparse",
                        metadata=document.metadata | {"text": document.text},
                    )
                )

        scored.sort(key=lambda candidate: (-candidate.score, candidate.card_id))
        return scored[:top_k]

