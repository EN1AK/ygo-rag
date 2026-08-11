from rag_agent.card_metadata import (
    TYPE_EFFECT,
    TYPE_LINK,
    TYPE_MONSTER,
    TYPE_PENDULUM,
    TYPE_XYZ,
    decode_effect_category_keys,
    decode_effect_category_labels,
    decode_card_structure,
    get_effect_category_labels,
)
from rag_agent.cards import Card, card_to_document


def test_decode_xyz_dark_monster_uses_rank():
    structure = decode_card_structure(
        type_value=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
        attribute_value=0x20,
        race_value=0x2000,
        level_value=4,
    )

    assert structure.card_kind == "monster"
    assert "xyz" in structure.monster_types
    assert structure.attribute == "dark"
    assert structure.race == "dragon"
    assert structure.rank == 4
    assert structure.level is None


def test_decode_non_xyz_monster_uses_level():
    structure = decode_card_structure(
        type_value=TYPE_MONSTER | TYPE_EFFECT,
        attribute_value=0x10,
        level_value=4,
    )

    assert structure.card_kind == "monster"
    assert structure.attribute == "light"
    assert structure.level == 4
    assert structure.rank is None


def test_decode_link_monster_uses_link_rating():
    structure = decode_card_structure(
        type_value=TYPE_MONSTER | TYPE_EFFECT | TYPE_LINK,
        attribute_value=0x08,
        level_value=2,
        atk_value=-2,
        defense_value=0x080 | 0x002,
    )

    assert structure.card_kind == "monster"
    assert "link" in structure.monster_types
    assert structure.attribute == "wind"
    assert structure.link_rating == 2
    assert structure.link_markers == ("top", "bottom")
    assert structure.attack is None
    assert structure.defense is None


def test_decode_pendulum_scale_from_level_high_byte():
    structure = decode_card_structure(
        type_value=TYPE_MONSTER | TYPE_EFFECT | TYPE_PENDULUM,
        attribute_value=0x20,
        race_value=0x2000000,
        level_value=(8 << 24) | 4,
        atk_value=1500,
        defense_value=-2,
    )

    assert structure.card_kind == "monster"
    assert "pendulum" in structure.monster_types
    assert structure.level == 4
    assert structure.pendulum_scale == 8
    assert structure.race == "illusion"
    assert structure.attack == 1500
    assert structure.defense is None


def test_structure_metadata_includes_chroma_filterable_fields():
    xyz = decode_card_structure(
        type_value=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
        attribute_value=0x20,
        level_value=4,
    ).to_metadata()
    level_monster = decode_card_structure(
        type_value=TYPE_MONSTER | TYPE_EFFECT,
        attribute_value=0x10,
        level_value=4,
    ).to_metadata()

    assert xyz["card_kind"] == "monster"
    assert xyz["attribute_name"] == "dark"
    assert xyz["is_xyz"] is True
    assert xyz["is_link"] is False
    assert xyz["rank"] == 4
    assert xyz["decoded_level"] is None
    assert level_monster["is_xyz"] is False
    assert level_monster["decoded_level"] == 4
    assert level_monster["rank"] is None


def test_decode_effect_category_bitmask_to_tags():
    category = 0x2 | 0x40000 | 0x400000 | 0x1000000 | 0x2000000

    assert decode_effect_category_keys(category) == (
        "monster_destroy",
        "special_summon",
        "lp_damage",
        "destruction_protection",
        "effect_protection",
    )
    assert decode_effect_category_labels(category) == (
        "怪兽破坏",
        "特殊召唤",
        "LP伤害",
        "破坏耐性",
        "效果耐性",
    )


def test_effect_category_labels_can_be_loaded_from_strings_conf(tmp_path, monkeypatch):
    strings_conf = tmp_path / "strings.conf"
    strings_conf.write_text(
        "\n".join(
            [
                "!system 1100 魔陷破坏",
                "!system 1101 怪兽破坏",
                "!system 1122 LP伤害",
                "!system 1124 破坏耐性",
                "!system 1125 效果耐性",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_STRINGS_CONF", str(strings_conf))
    get_effect_category_labels.cache_clear()

    try:
        assert decode_effect_category_labels(0x3400002) == (
            "怪兽破坏",
            "LP伤害",
            "破坏耐性",
            "效果耐性",
        )
    finally:
        get_effect_category_labels.cache_clear()


def test_card_document_includes_effect_category_tags():
    document = card_to_document(
        Card(
            1,
            "Alpha",
            "Destroy a monster.",
            category=0x2 | 0x40000,
        )
    )

    assert "效果标签：怪兽破坏 特殊召唤" in document.page_content
    assert document.metadata["effect_category_keys"] == "monster_destroy,special_summon"
    assert document.metadata["effect_category_names"] == "怪兽破坏,特殊召唤"
    assert document.metadata["has_effect_monster_destroy"] is True
    assert document.metadata["has_effect_draw"] is False
