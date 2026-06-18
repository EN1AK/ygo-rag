from rag_agent.cards import Card
from rag_agent.retrieval.sparse import SparseCardRetriever


def test_sparse_retriever_boosts_exact_card_name_match():
    cards = [
        Card(1, "我身作盾", "支付1500基本分，使破坏怪兽的效果发动无效并破坏。"),
        Card(2, "旋风", "以场上1张魔法·陷阱卡为对象才能发动。那张卡破坏。"),
    ]
    retriever = SparseCardRetriever.from_cards(cards)

    results = retriever.search("有没有效果类似“我身作盾”的卡", top_k=2)

    assert results[0].card_id == 1
    assert results[0].metadata["name"] == "我身作盾"


def test_sparse_retriever_matches_effect_terms():
    cards = [
        Card(1, "我身作盾", "支付1500基本分，使破坏怪兽的效果发动无效并破坏。"),
        Card(2, "神之宣告", "支付一半基本分，把魔法·陷阱卡的发动或怪兽的召唤无效并破坏。"),
        Card(3, "强欲之壶", "从卡组抽2张卡。"),
    ]
    retriever = SparseCardRetriever.from_cards(cards)

    results = retriever.search("无效并破坏发动", top_k=2)

    assert [candidate.card_id for candidate in results] == [1, 2]

