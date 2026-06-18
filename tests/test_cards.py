from rag_agent.cards import Card, card_to_document, normalize_effect_text


def test_normalize_effect_text_collapses_whitespace_without_removing_chinese_text():
    text = "当对方发动效果时，\r\n  可以支付1500基本分。\n使其发动无效。"

    assert normalize_effect_text(text) == "当对方发动效果时， 可以支付1500基本分。 使其发动无效。"


def test_card_to_document_includes_name_effect_and_metadata():
    card = Card(
        card_id=1,
        name="我身作盾",
        description="当对方发动破坏场上怪兽的效果时，可以支付1500基本分使其发动无效并破坏。",
        type=2,
        race=0,
        attribute=0,
        atk=-2,
        defense=-2,
        level=0,
    )

    document = card_to_document(card)

    assert document.page_content.startswith("卡名：我身作盾")
    assert "效果：" in document.page_content
    assert document.metadata["card_id"] == 1
    assert document.metadata["name"] == "我身作盾"

