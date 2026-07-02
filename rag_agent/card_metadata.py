from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
