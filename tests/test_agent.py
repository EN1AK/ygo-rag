from rag_agent.agent import CardRagAgent, RetrievedCard, format_retrieval_answer
from rag_agent.cards import Card
from rag_agent.card_metadata import TYPE_EFFECT, TYPE_MONSTER, TYPE_XYZ
from rag_agent.retrieval.hybrid import Candidate


def test_format_retrieval_answer_includes_rank_source_text_and_reason():
    cards = [
        RetrievedCard(
            card_id=1,
            name="我身作盾",
            score=0.95,
            source_text="支付1500基本分，使破坏怪兽的效果发动无效并破坏。",
            reason="都能应对破坏怪兽的效果。",
        )
    ]

    answer = format_retrieval_answer("有没有效果类似“我身作盾”的卡", cards)

    assert "1. 我身作盾" in answer
    assert "ID: 1" in answer
    assert "原文：" in answer
    assert "相似原因：" in answer


class FakeDenseRetriever:
    def search(self, query, *, top_k=10):
        return [
            Candidate(
                card_id=2,
                score=0.9,
                source="dense",
                metadata={
                    "name": "神之宣告",
                    "desc": "支付一半基本分，把发动或召唤无效并破坏。",
                },
            )
        ]


class FakeReranker:
    def __init__(self):
        self.received_candidates = []

    def rerank(self, query, candidates):
        self.received_candidates = list(candidates)
        return [
            Candidate(
                card_id=candidate.card_id,
                score=10.0 - index,
                source="reranker",
                metadata=candidate.metadata,
            )
            for index, candidate in enumerate(reversed(candidates))
        ]


class FakeLlm:
    def __init__(self):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return "LLM回答：" + prompt[:20]


def test_card_rag_agent_combines_sparse_dense_reranker_and_llm():
    cards = [
        Card(1, "我身作盾", "支付1500基本分，使破坏怪兽的效果发动无效并破坏。"),
        Card(2, "神之宣告", "支付一半基本分，把发动或召唤无效并破坏。"),
    ]
    llm = FakeLlm()
    agent = CardRagAgent(
        cards,
        dense_retriever=FakeDenseRetriever(),
        reranker=FakeReranker(),
        llm=llm,
    )

    answer = agent.query("有没有效果类似“我身作盾”的卡", top_k=2)

    assert answer.startswith("LLM回答：")
    assert llm.prompt is not None
    assert "我身作盾" in llm.prompt
    assert "神之宣告" in llm.prompt


def test_card_rag_agent_returns_structured_answer_without_llm():
    cards = [
        Card(1, "我身作盾", "支付1500基本分，使破坏怪兽的效果发动无效并破坏。"),
        Card(2, "强欲之壶", "从卡组抽2张卡。"),
    ]
    agent = CardRagAgent(cards)

    answer = agent.query("有没有效果类似“我身作盾”的卡", top_k=1)

    assert "1. 我身作盾" in answer
    assert "原文：" in answer


def test_card_rag_agent_expands_query_with_referenced_card_effect():
    cards = [
        Card(1, "我身作盾", "支付1500基本分，使破坏怪兽的效果发动无效并破坏。"),
        Card(2, "神之宣告", "支付一半基本分，把发动或召唤无效并破坏。"),
        Card(3, "强欲之壶", "从卡组抽2张卡。"),
    ]
    agent = CardRagAgent(cards)

    retrieved = agent.retrieve("有没有效果类似“我身作盾”的卡", top_k=3)

    assert [card.card_id for card in retrieved[:2]] == [1, 2]


class MixedDenseRetriever:
    def search(self, query, *, top_k=10):
        return [
            Candidate(2, 0.99, "dense", {"name": "非超量", "desc": "除外墓地"}),
            Candidate(1, 0.9, "dense", {"name": "四阶超量", "desc": "除外墓地"}),
        ]


def test_structured_filters_prefilter_sparse_and_dense_candidates():
    cards = [
        Card(
            1,
            "四阶超量",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
            attribute=0x20,
            race=0x2000,
            level=4,
        ),
        Card(
            2,
            "非超量",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT,
            attribute=0x20,
            race=0x2000,
            level=4,
        ),
    ]
    agent = CardRagAgent(cards, dense_retriever=MixedDenseRetriever())

    retrieved = agent.retrieve("效果是除外对手墓地的卡的四星超量怪兽", top_k=5)

    assert [card.card_id for card in retrieved] == [1]
    assert agent.last_structured_query.filters.rank == 4
    assert agent.last_filter_diagnostics.applied
    assert agent.last_filter_diagnostics.total_candidates == 2
    assert agent.last_filter_diagnostics.filtered_candidates == 1


def test_structured_filters_limit_reranker_input():
    cards = [
        Card(
            1,
            "四阶超量",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
            attribute=0x20,
            race=0x2000,
            level=4,
        ),
        Card(
            2,
            "非超量",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT,
            attribute=0x20,
            race=0x2000,
            level=4,
        ),
    ]
    reranker = FakeReranker()
    agent = CardRagAgent(cards, dense_retriever=MixedDenseRetriever(), reranker=reranker)

    agent.retrieve("除外对手墓地的四星超量怪兽", top_k=5)

    assert [candidate.card_id for candidate in reranker.received_candidates] == [1]


def test_structured_filters_warn_when_no_cards_match():
    warnings = []
    cards = [
        Card(
            1,
            "四阶超量",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
            attribute=0x20,
            race=0x2000,
            level=4,
        )
    ]
    agent = CardRagAgent(cards, warning_callback=warnings.append)

    retrieved = agent.retrieve("除外墓地的七阶超量怪兽", top_k=5)

    assert retrieved == []
    assert warnings
    assert agent.last_filter_diagnostics.filtered_candidates == 0


class MetadataAwareDenseRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, top_k=10, metadata_filter=None):
        self.calls.append((query, top_k, metadata_filter))
        return [
            Candidate(1, 0.9, "dense", {"name": "四阶超量", "desc": "除外对手墓地"})
        ]


def test_structured_filters_pass_metadata_filter_to_dense_retriever():
    cards = [
        Card(
            index,
            f"四阶超量{index}",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
            attribute=0x20,
            race=0x2000,
            level=4,
        )
        for index in range(1, 121)
    ]
    dense = MetadataAwareDenseRetriever()
    agent = CardRagAgent(cards, dense_retriever=dense)

    agent.retrieve("效果是除外对手墓地卡的四星超量怪兽", top_k=10)

    assert dense.calls
    assert dense.calls[0][1] == 100
    assert dense.calls[0][2] == {
        "$and": [
            {"card_kind": {"$eq": "monster"}},
            {"rank": {"$eq": 4}},
            {"is_xyz": {"$eq": True}},
        ]
    }


class FailingMetadataDenseRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, top_k=10, metadata_filter=None):
        self.calls.append((top_k, metadata_filter))
        if metadata_filter is not None:
            raise RuntimeError("missing metadata field")
        return [
            Candidate(1, 0.9, "dense", {"name": "四阶超量", "desc": "除外对手墓地"})
        ]


def test_dense_metadata_filter_failure_falls_back_and_warns():
    warnings = []
    cards = [
        Card(
            1,
            "四阶超量",
            "除外对手墓地的卡",
            type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
            attribute=0x20,
            race=0x2000,
            level=4,
        )
    ]
    dense = FailingMetadataDenseRetriever()
    agent = CardRagAgent(cards, dense_retriever=dense, warning_callback=warnings.append)

    retrieved = agent.retrieve("效果是除外对手墓地卡的四星超量怪兽", top_k=10)

    assert [card.card_id for card in retrieved] == [1]
    assert dense.calls[0][1] is not None
    assert dense.calls[1] == (200, None)
    assert warnings
    assert "Dense metadata filtering failed" in warnings[0]


def test_structured_sparse_recall_pool_keeps_candidate_below_final_top_k_for_rerank():
    cards = []
    for index in range(1, 37):
        cards.append(
            Card(
                index,
                f"高字面匹配{index}",
                "除外对手墓地卡。除外对手墓地卡。",
                type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
                attribute=0x20,
                race=0x2000,
                level=4,
            )
        )
    cards.append(
        Card(
            80,
            "No.80 狂装霸王 狂想战曲王",
            "把这张卡1个超量素材取除，以对方墓地1张卡为对象才能发动。那张卡除外。",
            type=TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,
            attribute=0x20,
            race=0x2000,
            level=4,
        )
    )
    reranker = FakeReranker()
    agent = CardRagAgent(cards, reranker=reranker)

    agent.retrieve("效果是除外对手墓地卡的四星超量怪兽", top_k=10)

    assert 80 in [candidate.card_id for candidate in reranker.received_candidates]
    assert len(reranker.received_candidates) >= 37
