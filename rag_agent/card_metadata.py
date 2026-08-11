from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


TYPE_MONSTER = 0x1
TYPE_SPELL = 0x2
TYPE_TRAP = 0x4
TYPE_NORMAL = 0x10
TYPE_EFFECT = 0x20
TYPE_FUSION = 0x40
TYPE_RITUAL = 0x80
TYPE_TRAPMONSTER = 0x100
TYPE_SPIRIT = 0x200
TYPE_UNION = 0x400
TYPE_GEMINI = 0x800
TYPE_TUNER = 0x1000
TYPE_SYNCHRO = 0x2000
TYPE_TOKEN = 0x4000
TYPE_QUICKPLAY = 0x10000
TYPE_CONTINUOUS = 0x20000
TYPE_EQUIP = 0x40000
TYPE_FIELD = 0x80000
TYPE_COUNTER = 0x100000
TYPE_FLIP = 0x200000
TYPE_TOON = 0x400000
TYPE_XYZ = 0x800000
TYPE_PENDULUM = 0x1000000
TYPE_SPSUMMON = 0x2000000
TYPE_LINK = 0x4000000

ATTRIBUTE_NAMES = {
    0x01: "earth",
    0x02: "water",
    0x04: "fire",
    0x08: "wind",
    0x10: "light",
    0x20: "dark",
    0x40: "divine",
}

RACE_NAMES = {
    0x1: "warrior",
    0x2: "spellcaster",
    0x4: "fairy",
    0x8: "fiend",
    0x10: "zombie",
    0x20: "machine",
    0x40: "aqua",
    0x80: "pyro",
    0x100: "rock",
    0x200: "winged_beast",
    0x400: "plant",
    0x800: "insect",
    0x1000: "thunder",
    0x2000: "dragon",
    0x4000: "beast",
    0x8000: "beast_warrior",
    0x10000: "dinosaur",
    0x20000: "fish",
    0x40000: "sea_serpent",
    0x80000: "reptile",
    0x100000: "psychic",
    0x200000: "divine_beast",
    0x400000: "creator_god",
    0x800000: "wyrm",
    0x1000000: "cyberse",
    0x2000000: "illusion",
}

TYPE_NAMES = {
    TYPE_MONSTER: "monster",
    TYPE_SPELL: "spell",
    TYPE_TRAP: "trap",
    TYPE_NORMAL: "normal",
    TYPE_EFFECT: "effect",
    TYPE_FUSION: "fusion",
    TYPE_RITUAL: "ritual",
    TYPE_TRAPMONSTER: "trap_monster",
    TYPE_SPIRIT: "spirit",
    TYPE_UNION: "union",
    TYPE_GEMINI: "gemini",
    TYPE_TUNER: "tuner",
    TYPE_SYNCHRO: "synchro",
    TYPE_TOKEN: "token",
    TYPE_QUICKPLAY: "quickplay",
    TYPE_CONTINUOUS: "continuous",
    TYPE_EQUIP: "equip",
    TYPE_FIELD: "field",
    TYPE_COUNTER: "counter",
    TYPE_FLIP: "flip",
    TYPE_TOON: "toon",
    TYPE_XYZ: "xyz",
    TYPE_PENDULUM: "pendulum",
    TYPE_SPSUMMON: "special_summon",
    TYPE_LINK: "link",
}

LINK_MARKERS = {
    0x040: "top_left",
    0x080: "top",
    0x100: "top_right",
    0x008: "left",
    0x020: "right",
    0x001: "bottom_left",
    0x002: "bottom",
    0x004: "bottom_right",
}

EFFECT_CATEGORY_NAMES = {
    0x1: "spell_trap_destroy",
    0x2: "monster_destroy",
    0x4: "banish",
    0x8: "to_grave",
    0x10: "to_hand",
    0x20: "to_deck",
    0x40: "hand_destroy",
    0x80: "deck_destroy",
    0x100: "draw",
    0x200: "search",
    0x400: "card_recover",
    0x800: "battle_position",
    0x1000: "control",
    0x2000: "atk_def_change",
    0x4000: "piercing_damage",
    0x8000: "multi_attack",
    0x10000: "attack_limit",
    0x20000: "direct_attack",
    0x40000: "special_summon",
    0x80000: "token",
    0x100000: "race_related",
    0x200000: "attribute_related",
    0x400000: "lp_damage",
    0x800000: "lp_recover",
    0x1000000: "destruction_protection",
    0x2000000: "effect_protection",
    0x4000000: "counter",
    0x8000000: "luck",
    0x10000000: "fusion_related",
    0x20000000: "synchro_related",
    0x40000000: "xyz_related",
    0x80000000: "negate_effect",
}

DEFAULT_EFFECT_CATEGORY_LABELS = {
    0x1: "魔陷破坏",
    0x2: "怪兽破坏",
    0x4: "除外",
    0x8: "送去墓地",
    0x10: "返回手卡",
    0x20: "返回卡组",
    0x40: "手卡破坏",
    0x80: "卡组破坏",
    0x100: "抽卡辅助",
    0x200: "卡组检索",
    0x400: "卡片回收",
    0x800: "表示形式",
    0x1000: "控制权",
    0x2000: "攻守变化",
    0x4000: "穿刺伤害",
    0x8000: "多次攻击",
    0x10000: "攻击限制",
    0x20000: "直接攻击",
    0x40000: "特殊召唤",
    0x80000: "衍生物",
    0x100000: "种族相关",
    0x200000: "属性相关",
    0x400000: "LP伤害",
    0x800000: "LP回复",
    0x1000000: "破坏耐性",
    0x2000000: "效果耐性",
    0x4000000: "指示物",
    0x8000000: "幸运",
    0x10000000: "融合相关",
    0x20000000: "同调相关",
    0x40000000: "超量相关",
    0x80000000: "效果无效",
}

_STRINGS_CONF_ENV_VARS = ("RAG_STRINGS_CONF", "YGOPRO_STRINGS_CONF")
_DEFAULT_STRINGS_CONF_PATHS = (
    Path("data/strings.conf"),
    Path("strings.conf"),
    Path("assets/strings.conf"),
    Path("asset/strings.conf"),
)


@dataclass(frozen=True)
class CardStructure:
    card_kind: str | None
    monster_types: tuple[str, ...] = ()
    attribute: str | None = None
    race: str | None = None
    level: int | None = None
    rank: int | None = None
    link_rating: int | None = None
    pendulum_scale: int | None = None
    link_markers: tuple[str, ...] = ()
    attack: int | None = None
    defense: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "card_kind": self.card_kind,
            "monster_types": ",".join(self.monster_types) if self.monster_types else None,
            "type_names": ",".join(self.type_names) if self.type_names else None,
            "attribute_name": self.attribute,
            "race_name": self.race,
            "decoded_level": self.level,
            "rank": self.rank,
            "link_rating": self.link_rating,
            "pendulum_scale": self.pendulum_scale,
            "link_markers": ",".join(self.link_markers) if self.link_markers else None,
            "decoded_atk": self.attack,
            "decoded_def": self.defense,
        }
        for monster_type in (
            "normal",
            "effect",
            "fusion",
            "ritual",
            "synchro",
            "xyz",
            "pendulum",
            "link",
            "tuner",
            "flip",
            "spirit",
            "union",
            "gemini",
            "toon",
            "token",
            "special_summon",
        ):
            metadata[f"is_{monster_type}"] = monster_type in self.monster_types
        return metadata

    @property
    def type_names(self) -> tuple[str, ...]:
        names = list(self.monster_types)
        if self.card_kind in {"spell", "trap"}:
            names.insert(0, self.card_kind)
        return tuple(dict.fromkeys(names))


def decode_card_structure(
    *,
    type_value: int | None,
    attribute_value: int | None = None,
    race_value: int | None = None,
    level_value: int | None = None,
    atk_value: int | None = None,
    defense_value: int | None = None,
) -> CardStructure:
    type_bits = int(type_value or 0)
    if type_bits & TYPE_MONSTER:
        card_kind = "monster"
    elif type_bits & TYPE_SPELL:
        card_kind = "spell"
    elif type_bits & TYPE_TRAP:
        card_kind = "trap"
    else:
        card_kind = None

    monster_types: list[str] = []
    if card_kind == "monster":
        for bit, name in (
            (TYPE_NORMAL, "normal"),
            (TYPE_EFFECT, "effect"),
            (TYPE_FUSION, "fusion"),
            (TYPE_RITUAL, "ritual"),
            (TYPE_TRAPMONSTER, "trap_monster"),
            (TYPE_SPIRIT, "spirit"),
            (TYPE_UNION, "union"),
            (TYPE_GEMINI, "gemini"),
            (TYPE_TUNER, "tuner"),
            (TYPE_SYNCHRO, "synchro"),
            (TYPE_TOKEN, "token"),
            (TYPE_FLIP, "flip"),
            (TYPE_TOON, "toon"),
            (TYPE_XYZ, "xyz"),
            (TYPE_PENDULUM, "pendulum"),
            (TYPE_SPSUMMON, "special_summon"),
            (TYPE_LINK, "link"),
        ):
            if type_bits & bit:
                monster_types.append(name)

    numeric_value = _decode_level_value(level_value)
    level = rank = link_rating = None
    if card_kind == "monster":
        if "xyz" in monster_types:
            rank = numeric_value
        elif "link" in monster_types:
            link_rating = numeric_value
        else:
            level = numeric_value

    return CardStructure(
        card_kind=card_kind,
        monster_types=tuple(monster_types),
        attribute=ATTRIBUTE_NAMES.get(int(attribute_value or 0)),
        race=RACE_NAMES.get(int(race_value or 0)),
        level=level,
        rank=rank,
        link_rating=link_rating,
        pendulum_scale=_decode_pendulum_scale(level_value)
        if type_bits & TYPE_PENDULUM
        else None,
        link_markers=tuple(_decode_link_markers(defense_value))
        if type_bits & TYPE_LINK
        else (),
        attack=_decode_stat_value(atk_value) if card_kind == "monster" else None,
        defense=(
            None
            if card_kind != "monster" or type_bits & TYPE_LINK
            else _decode_stat_value(defense_value)
        ),
    )


def decode_effect_category_keys(category_value: int | None) -> tuple[str, ...]:
    category_bits = int(category_value or 0)
    return tuple(
        name
        for bit, name in EFFECT_CATEGORY_NAMES.items()
        if category_bits & bit
    )


def decode_effect_category_labels(
    category_value: int | None,
    labels_by_bit: Mapping[int, str] | None = None,
) -> tuple[str, ...]:
    labels = labels_by_bit or get_effect_category_labels()
    category_bits = int(category_value or 0)
    return tuple(
        label
        for bit, label in _ordered_effect_category_labels(labels).items()
        if category_bits & bit and label
    )


def decode_effect_categories(category_value: int | None) -> tuple[str, ...]:
    return decode_effect_category_labels(category_value)


def load_effect_category_labels_from_strings_conf(path: Path | str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3 or parts[0] != "!system":
            continue
        try:
            system_id = int(parts[1])
        except ValueError:
            continue
        offset = system_id - 1100
        if 0 <= offset < len(EFFECT_CATEGORY_NAMES):
            labels[1 << offset] = parts[2].strip()
    return labels


@lru_cache(maxsize=1)
def get_effect_category_labels() -> dict[int, str]:
    strings_conf = find_strings_conf()
    if strings_conf is None:
        return dict(DEFAULT_EFFECT_CATEGORY_LABELS)
    labels = load_effect_category_labels_from_strings_conf(strings_conf)
    return dict(DEFAULT_EFFECT_CATEGORY_LABELS) | labels


def find_strings_conf() -> Path | None:
    candidates: list[Path] = []
    for env_var in _STRINGS_CONF_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))
    candidates.extend(_DEFAULT_STRINGS_CONF_PATHS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ordered_effect_category_labels(labels: Mapping[int, str]) -> dict[int, str]:
    return {
        bit: labels[bit]
        for bit in EFFECT_CATEGORY_NAMES
        if bit in labels
    }


def effect_categories_to_metadata(category_value: int | None) -> dict[str, Any]:
    keys = decode_effect_category_keys(category_value)
    labels = decode_effect_category_labels(category_value)
    metadata: dict[str, Any] = {
        "effect_category_keys": ",".join(keys) if keys else None,
        "effect_category_names": ",".join(labels) if labels else None,
    }
    for key in EFFECT_CATEGORY_NAMES.values():
        metadata[f"has_effect_{key}"] = key in keys
    return metadata


def _decode_level_value(value: int | None) -> int | None:
    if value is None:
        return None
    raw = int(value)
    if raw <= 0:
        return None
    return raw & 0xFF


def _decode_pendulum_scale(value: int | None) -> int | None:
    if value is None:
        return None
    raw = int(value)
    scale = (raw >> 24) & 0xFF
    return scale if scale > 0 else None


def _decode_link_markers(value: int | None) -> list[str]:
    raw = int(value or 0)
    return [name for bit, name in LINK_MARKERS.items() if raw & bit]


def _decode_stat_value(value: int | None) -> int | None:
    if value is None:
        return None
    raw = int(value)
    return None if raw == -2 else raw
