from pathlib import Path

from rag_agent.retrieval.vector import (
    ChromaConfig,
    ChromaVectorStore,
    structured_filters_to_chroma_where,
)
from rag_agent.structured_query import StructuredQueryFilters


def test_structured_filters_to_chroma_where_for_rank_four_xyz():
    where = structured_filters_to_chroma_where(
        StructuredQueryFilters(
            card_kind="monster",
            monster_types=("xyz",),
            rank=4,
            attribute="dark",
        )
    )

    assert where == {
        "$and": [
            {"card_kind": {"$eq": "monster"}},
            {"attribute_name": {"$eq": "dark"}},
            {"rank": {"$eq": 4}},
            {"is_xyz": {"$eq": True}},
        ]
    }


def test_structured_filters_to_chroma_where_returns_none_for_empty_filters():
    assert structured_filters_to_chroma_where(StructuredQueryFilters()) is None


def test_chroma_vector_store_forwards_metadata_filter():
    class FakeDocument:
        page_content = "卡名：测试\n效果：测试"
        metadata = {"card_id": 1, "name": "测试", "desc": "测试"}

    class FakeStore:
        def __init__(self):
            self.calls = []

        def similarity_search_with_score(self, query, *, k, **kwargs):
            self.calls.append((query, k, kwargs))
            return [(FakeDocument(), 0.25)]

    class TestVectorStore(ChromaVectorStore):
        def __init__(self):
            super().__init__(ChromaConfig(Path("unused")), embedding_function=object())
            self.fake_store = FakeStore()

        def load(self):
            return self.fake_store

    vector_store = TestVectorStore()
    where = {"rank": {"$eq": 4}}

    results = vector_store.search("除外墓地", top_k=5, metadata_filter=where)

    assert vector_store.fake_store.calls == [("除外墓地", 5, {"filter": where})]
    assert [candidate.card_id for candidate in results] == [1]
