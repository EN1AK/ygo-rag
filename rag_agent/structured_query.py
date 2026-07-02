from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag_agent.card_metadata import decode_card_structure
from rag_agent.cards import Card


_CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_ATTRIBUTES = {
    "暗属性": "dark",
    "暗屬性": "dark",
    "光属性": "light",
    "光屬性": "light",
    "地属性": "earth",
    "地屬性": "earth",
    "水属性": "water",
    "水屬性": "water",
    "炎属性": "fire",
    "炎屬性": "fire",
    "火属性": "fire",
    "火屬性": "fire",
    "风属性": "wind",
    "風屬性": "wind",
    "神属性": "divine",
    "神屬性": "divine",
}

_MONSTER_TYPES = {
    "超量": "xyz",
    "XYZ": "xyz",
    "xyz": "xyz",
    "融合": "fusion",
    "同调": "synchro",
    "同步": "synchro",
    "连接": "link",
    "链接": "link",
    "LINK": "link",
    "link": "link",
    "仪式": "ritual",
    "效果怪兽": "effect",
    "效果怪獸": "effect",
    "通常怪兽": "normal",
    "通常怪獸": "normal",
}

_CARD_KINDS = {
    "怪兽": "monster",
    "怪獸": "monster",
    "魔法": "spell",
    "陷阱": "trap",
}


@dataclass(frozen=True)
class StructuredQueryFilters:
    card_kind: str | None = None
    monster_types: tuple[str, ...] = ()
    attribute: str | None = None
    race: str | None = None
    level: int | None = None
    rank: int | None = None
    link_rating: int | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.card_kind,
                self.monster_types,
                self.attribute,
                self.race,
                self.level is not None,
                self.rank is not None,
                self.link_rating is not None,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_kind": self.card_kind,
            "monster_types": list(self.monster_types),
            "attribute": self.attribute,
            "race": self.race,
            "level": self.level,
            "rank": self.rank,
            "link_rating": self.link_rating,
        }


@dataclass(frozen=True)
class StructuredQuery:
    original_query: str
    effect_query: str
    filters: StructuredQueryFilters
    matched_terms: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def has_filters(self) -> bool:
        return not self.filters.is_empty()

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "effect_query": self.effect_query,
            "filters": self.filters.to_dict(),
            "has_filters": self.has_filters,
            "matched_terms": list(self.matched_terms),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class FilterDiagnostics:
    applied: bool = False
    total_candidates: int = 0
    filtered_candidates: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "total_candidates": self.total_candidates,
            "filtered_candidates": self.filtered_candidates,
            "warnings": list(self.warnings),
        }


def parse_structured_query(query: str) -> StructuredQuery:
    filters = _MutableFilters()
    matched_terms: list[str] = []
    effect_query = query

    for term, attribute in _ATTRIBUTES.items():
        if term in query:
            filters.attribute = attribute
            matched_terms.append(term)
            effect_query = effect_query.replace(term, "")

    for term, card_kind in _CARD_KINDS.items():
        if term in query:
            filters.card_kind = card_kind
            matched_terms.append(term)
            effect_query = effect_query.replace(term, "")

    for term, monster_type in _MONSTER_TYPES.items():
        if term in query:
            filters.monster_types.add(monster_type)
            if monster_type in {"xyz", "fusion", "synchro", "link", "ritual", "effect", "normal"}:
                filters.card_kind = "monster"
            matched_terms.append(term)
            effect_query = effect_query.replace(term, "")

    for match in re.finditer(r"([0-9０-９一二两三四五六七八九十]+)\s*(星|阶|階)", query):
        number = _parse_number(match.group(1))
        if number is None:
            continue
        term = match.group(0)
        matched_terms.append(term)
        effect_query = effect_query.replace(term, "")
        filters.card_kind = "monster"
        if match.group(2) in {"阶", "階"} or "xyz" in filters.monster_types:
            filters.rank = number
        else:
            filters.level = number

    for match in re.finditer(r"(?:rank|Rank|RANK)\s*[- ]?\s*([0-9０-９一二两三四五六七八九十]+)", query):
        number = _parse_number(match.group(1))
        if number is None:
            continue
        matched_terms.append(match.group(0))
        effect_query = effect_query.replace(match.group(0), "")
        filters.card_kind = "monster"
        filters.monster_types.add("xyz")
        filters.rank = number

    for match in re.finditer(r"(?:link|Link|LINK|连接|链接)\s*[- ]?\s*([0-9０-９一二两三四五六七八九十]+)", query):
        number = _parse_number(match.group(1))
        if number is None:
            continue
        matched_terms.append(match.group(0))
        effect_query = effect_query.replace(match.group(0), "")
        filters.card_kind = "monster"
        filters.monster_types.add("link")
        filters.link_rating = number

    if "xyz" in filters.monster_types and filters.level is not None and filters.rank is None:
        filters.rank = filters.level
        filters.level = None

    normalized_effect_query = _normalize_effect_query(effect_query)
    if not normalized_effect_query:
        normalized_effect_query = query

    return StructuredQuery(
        original_query=query,
        effect_query=normalized_effect_query,
        filters=filters.freeze(),
        matched_terms=tuple(dict.fromkeys(matched_terms)),
    )


def card_matches_filters(card: Card, filters: StructuredQueryFilters) -> bool:
    if filters.is_empty():
        return True
    structure = decode_card_structure(
        type_value=card.type,
        attribute_value=card.attribute,
        race_value=card.race,
        level_value=card.level,
    )
    if filters.card_kind and structure.card_kind != filters.card_kind:
        return False
    for monster_type in filters.monster_types:
        if monster_type not in structure.monster_types:
            return False
    if filters.attribute and structure.attribute != filters.attribute:
        return False
    if filters.race and structure.race != filters.race:
        return False
    if filters.level is not None and structure.level != filters.level:
        return False
    if filters.rank is not None and structure.rank != filters.rank:
        return False
    if filters.link_rating is not None and structure.link_rating != filters.link_rating:
        return False
    return True


@dataclass
class _MutableFilters:
    card_kind: str | None = None
    monster_types: set[str] = field(default_factory=set)
    attribute: str | None = None
    race: str | None = None
    level: int | None = None
    rank: int | None = None
    link_rating: int | None = None

    def freeze(self) -> StructuredQueryFilters:
        return StructuredQueryFilters(
            card_kind=self.card_kind,
            monster_types=tuple(sorted(self.monster_types)),
            attribute=self.attribute,
            race=self.race,
            level=self.level,
            rank=self.rank,
            link_rating=self.link_rating,
        )


def _parse_number(value: str) -> int | None:
    normalized = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if normalized.isdigit():
        return int(normalized)
    if normalized in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[normalized]
    if normalized.startswith("十") and len(normalized) == 2:
        suffix = _CHINESE_DIGITS.get(normalized[1])
        return 10 + suffix if suffix is not None else None
    if normalized.endswith("十") and len(normalized) == 2:
        prefix = _CHINESE_DIGITS.get(normalized[0])
        return prefix * 10 if prefix is not None else None
    if "十" in normalized and len(normalized) == 3:
        prefix = _CHINESE_DIGITS.get(normalized[0])
        suffix = _CHINESE_DIGITS.get(normalized[2])
        if prefix is not None and suffix is not None:
            return prefix * 10 + suffix
    return None


def _normalize_effect_query(query: str) -> str:
    normalized = re.sub(r"效果是|效果为|效果：|效果:", "", query)
    normalized = re.sub(r"有没有|有沒有|类似|類似|的卡|且是|并且是|而且是", "", normalized)
    normalized = re.sub(r"^[的是为：:]+", "", normalized)
    normalized = re.sub(r"[的是为：:]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" ，,。?？：:")
    return normalized.strip()
