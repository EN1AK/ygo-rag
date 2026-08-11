from rag_agent.query_service import (
    QueryResponse,
    build_structured_response,
    truncate_text,
)


def sample_response():
    return QueryResponse(
        answer="相近效果候选...",
        results=[
            {
                "card_id": 65192027,
                "name": "暗黑武装龙",
                "score": 0.95,
                "source_text": "自己墓地的暗属性怪兽是3只的场合才能特殊召唤。除外暗属性怪兽破坏场上1张卡。",
                "reason": "暗属性、墓地资源、破坏场上卡的方向相近。",
            }
        ],
        warnings=["测试 warning"],
    )


def test_build_structured_response_creates_one_card_block_per_result():
    structured = build_structured_response(sample_response())

    assert structured["version"] == 1
    assert structured["summary"]["result_count"] == 1
    assert structured["summary"]["warnings"] == ["测试 warning"]
    assert len(structured["blocks"]) == 1
    block = structured["blocks"][0]
    assert block["type"] == "card"
    assert block["index"] == 1
    assert block["card_id"] == 65192027
    assert block["name"] == "暗黑武装龙"
    assert block["score"] == 0.95
    assert "1. 暗黑武装龙" in block["text"]
    assert "ID: 65192027" in block["text"]
    assert block["fields"]["source_text"].startswith("自己墓地")
    assert block["fields"]["reason"] == "暗属性、墓地资源、破坏场上卡的方向相近。"
    assert block["truncated"] is False


def test_structured_response_truncates_card_block_text():
    structured = build_structured_response(sample_response(), max_block_chars=80)

    block = structured["blocks"][0]
    assert len(block["text"]) <= 80
    assert block["text"].endswith("…")
    assert block["truncated"] is True
    assert block["fields"]["source_text"].startswith("自己墓地")


def test_truncate_text_handles_tiny_limits():
    assert truncate_text("abcdef", 1) == "…"
    assert truncate_text("abcdef", 3) == "ab…"
    assert truncate_text("abc", 3) == "abc"
