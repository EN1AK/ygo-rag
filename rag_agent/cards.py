from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    description: str
    type: int | None = None
    race: int | None = None
    attribute: int | None = None
    atk: int | None = None
    defense: int | None = None
    level: int | None = None
    category: int | None = None


@dataclass(frozen=True)
class RetrievalDocument:
    page_content: str
    metadata: dict[str, Any]


def normalize_effect_text(text: str) -> str:
    """Normalize text for retrieval while preserving readable punctuation."""
    return " ".join(text.replace("\r\n", "\n").replace("\n", " ").split())


def card_to_document(card: Card) -> RetrievalDocument:
    description = normalize_effect_text(card.description)
    page_content = f"卡名：{card.name}\n效果：{description}"
    metadata = {
        "card_id": card.card_id,
        "name": card.name,
        "desc": card.description,
        "type": card.type,
        "race": card.race,
        "attribute": card.attribute,
        "atk": card.atk,
        "def": card.defense,
        "level": card.level,
        "category": card.category,
    }
    return RetrievalDocument(page_content=page_content, metadata=metadata)

