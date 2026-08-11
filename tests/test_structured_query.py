from rag_agent.card_metadata import TYPE_EFFECT, TYPE_MONSTER, TYPE_XYZ
from rag_agent.cards import Card
from rag_agent.structured_query import card_matches_filters, parse_structured_query


def test_parse_rank_four_xyz_monster_query_and_effect_text():
    parsed = parse_structured_query("效果是除外或者回收对手墓地的卡的四星超量怪兽")

    assert parsed.filters.card_kind == "monster"
    assert "xyz" in parsed.filters.monster_types
    assert parsed.filters.rank == 4
    assert parsed.filters.level is None
    assert parsed.effect_query == "除外或者回收对手墓地"


def test_parse_dark_attribute_query():
    parsed = parse_structured_query("有没有效果类似暗黑武装龙且是暗属性的卡")

    assert parsed.filters.attribute == "dark"
    assert parsed.has_filters


def test_parse_no_filter_query():
    parsed = parse_structured_query("有没有能无效并破坏发动的卡")

    assert not parsed.has_filters
    assert parsed.effect_query == "能无效并破坏发动"


def test_parse_link_rating():
    parsed = parse_structured_query("可以除外墓地卡的LINK-2连接怪兽")

    assert parsed.filters.card_kind == "monster"
    assert "link" in parsed.filters.monster_types
    assert parsed.filters.link_rating == 2


def test_card_matches_filters_for_rank_four_xyz():
    parsed = parse_structured_query("除外墓地的四阶超量怪兽")
    matching = Card(
        1,
        "测试超量",
        "除外墓地",
        type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
        attribute=0x20,
        race=0x2000,
        level=4,
    )
    non_matching = Card(
        2,
        "测试效果怪兽",
        "除外墓地",
        type=TYPE_MONSTER | TYPE_EFFECT,
        attribute=0x20,
        race=0x2000,
        level=4,
    )

    assert card_matches_filters(matching, parsed.filters)
    assert not card_matches_filters(non_matching, parsed.filters)
