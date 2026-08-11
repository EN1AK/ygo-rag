import pytest

from rag_agent.retrieval.hybrid import Candidate
from rag_agent.retrieval.reranker import (
    LlmReranker,
    build_llm_rerank_prompt,
    parse_llm_rerank_response,
)


class FakeLlm:
    def __init__(self, content):
        self.content = content
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return self.content


def candidates():
    return [
        Candidate(
            1,
            0.3,
            "hybrid",
            {"name": "我身作盾", "desc": "支付1500基本分，使破坏效果无效并破坏。"},
        ),
        Candidate(
            2,
            0.2,
            "hybrid",
            {"name": "神之密告", "desc": "支付1500基本分，使魔法·陷阱卡发动无效并破坏。"},
        ),
        Candidate(
            3,
            0.1,
            "hybrid",
            {"name": "星光大道", "desc": "要把自己场上的卡破坏的效果发动时无效并破坏。"},
        ),
    ]


def test_build_llm_rerank_prompt_includes_candidate_ids_names_text_and_scores():
    prompt = build_llm_rerank_prompt("有没有效果类似“我身作盾”的卡", candidates()[:2])

    assert "只允许评价候选列表中的卡" in prompt
    assert "card_id" in prompt
    assert "我身作盾" in prompt
    assert "神之密告" in prompt
    assert "0.3000" in prompt
    assert "支付1500基本分" in prompt


def test_parse_llm_rerank_response_accepts_structured_json():
    parsed = parse_llm_rerank_response(
        '{"results":[{"card_id":3,"score":0.95,"reason":"保护自己场上的卡"},'
        '{"card_id":1,"score":0.90,"reason":"参考卡本身"}]}',
        allowed_ids={1, 2, 3},
    )

    assert [item.card_id for item in parsed] == [3, 1]
    assert parsed[0].score == 0.95
    assert parsed[0].reason == "保护自己场上的卡"


def test_parse_llm_rerank_response_rejects_unknown_candidate_ids():
    parsed = parse_llm_rerank_response(
        '{"results":[{"card_id":999,"score":1.0,"reason":"编造"},'
        '{"card_id":1,"score":0.5,"reason":"有效"}]}',
        allowed_ids={1},
    )

    assert [item.card_id for item in parsed] == [1]


def test_llm_reranker_preserves_unscored_candidates_after_scored_candidates():
    llm = FakeLlm(
        '{"results":[{"card_id":3,"score":0.95,"reason":"更像保护效果"},'
        '{"card_id":1,"score":0.80,"reason":"参考卡"}]}'
    )
    reranker = LlmReranker(llm=llm)

    reranked = reranker.rerank("有没有效果类似“我身作盾”的卡", candidates())

    assert [candidate.card_id for candidate in reranked] == [3, 1, 2]
    assert reranked[0].score == 0.95
    assert reranked[0].source == "llm_reranker"
    assert reranked[0].metadata["llm_judge_reason"] == "更像保护效果"
    assert reranked[2].metadata["llm_judge_reason"] == "LLM judge did not score this candidate."


def test_llm_reranker_limits_candidates_sent_to_llm():
    llm = FakeLlm('{"results":[{"card_id":1,"score":1.0,"reason":"只评第一个"}]}')
    reranker = LlmReranker(llm=llm, max_candidates=1)

    reranked = reranker.rerank("query", candidates())

    assert '"card_id": 1' in llm.prompt
    assert '"card_id": 2' not in llm.prompt
    assert [candidate.card_id for candidate in reranked] == [1, 2, 3]


def test_parse_llm_rerank_response_reports_malformed_json():
    with pytest.raises(RuntimeError, match="Unable to parse LLM rerank response"):
        parse_llm_rerank_response("不是 JSON", allowed_ids={1})


def test_llm_reranker_falls_back_to_original_candidates_on_timeout():
    warnings = []

    class TimeoutLlm:
        def invoke(self, prompt):
            raise TimeoutError("request timed out")

    reranker = LlmReranker(llm=TimeoutLlm(), warning_callback=warnings.append)

    reranked = reranker.rerank("query", candidates()[:2])

    assert [candidate.card_id for candidate in reranked] == [1, 2]
    assert reranked[0].source == "hybrid"
    assert warnings
    assert "LLM rerank failed" in warnings[0]


def test_llm_reranker_falls_back_to_original_candidates_on_malformed_json():
    warnings = []
    reranker = LlmReranker(llm=FakeLlm("不是 JSON"), warning_callback=warnings.append)

    reranked = reranker.rerank("query", candidates()[:2])

    assert [candidate.card_id for candidate in reranked] == [1, 2]
    assert warnings
    assert "Unable to parse LLM rerank response" in warnings[0]
