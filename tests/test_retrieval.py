from rag_agent.retrieval.hybrid import Candidate, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_merges_and_deduplicates_candidates():
    dense = [
        Candidate(card_id=1, score=0.90, source="dense", metadata={"name": "A"}),
        Candidate(card_id=2, score=0.80, source="dense", metadata={"name": "B"}),
    ]
    sparse = [
        Candidate(card_id=2, score=10.0, source="sparse", metadata={"name": "B"}),
        Candidate(card_id=3, score=9.0, source="sparse", metadata={"name": "C"}),
    ]

    results = reciprocal_rank_fusion([dense, sparse], k=60)

    assert [candidate.card_id for candidate in results] == [2, 1, 3]
    assert results[0].metadata["name"] == "B"
    assert results[0].source == "hybrid"


def test_reciprocal_rank_fusion_limits_top_k():
    dense = [
        Candidate(card_id=1, score=0.90, source="dense", metadata={}),
        Candidate(card_id=2, score=0.80, source="dense", metadata={}),
        Candidate(card_id=3, score=0.70, source="dense", metadata={}),
    ]

    results = reciprocal_rank_fusion([dense], top_k=2)

    assert [candidate.card_id for candidate in results] == [1, 2]


def test_reciprocal_rank_fusion_uses_original_score_as_tie_breaker():
    dense = [
        Candidate(card_id=1, score=0.9, source="dense", metadata={"name": "unrelated"}),
    ]
    sparse = [
        Candidate(card_id=2, score=1000.0, source="sparse", metadata={"name": "exact"}),
    ]

    results = reciprocal_rank_fusion([dense, sparse])

    assert [candidate.card_id for candidate in results] == [2, 1]
