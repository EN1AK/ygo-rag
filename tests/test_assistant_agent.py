import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_agent.assistant_agent import (
    ASSISTANT_TOOL_SCHEMAS,
    AssistantAgent,
    AssistantAgentRequest,
    AssistantToolObservation,
    ProjectAssistantToolRunner,
    build_assistant_planner_prompt,
    parse_assistant_planner_action,
)
from rag_agent.adjudication import (
    AdjudicationResponse,
    ParsedScenario,
    SimulationStatus,
)
from rag_agent.config import Settings
from rag_agent.query_service import QueryResponse
from rag_agent.translation_service import build_translation_response


class FakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.responses.pop(0))


class FakeToolRunner:
    def __init__(self):
        self.calls = []

    def run(self, tool, arguments, request):
        self.calls.append((tool, dict(arguments), request))
        if tool == "search_cards":
            return AssistantToolObservation(
                tool=tool,
                ok=True,
                summary="查卡完成，返回 1 张候选卡。",
                data={
                    "answer": "候选：我身作盾",
                    "results": [
                        {
                            "card_id": 1,
                            "name": "我身作盾",
                            "score": 1.0,
                            "source_text": "支付1500基本分，使发动无效并破坏。",
                        }
                    ],
                },
            )
        if tool == "translate_text":
            return AssistantToolObservation(
                tool=tool,
                ok=True,
                summary="翻译完成：ja -> zh-CN。",
                data={"translation": "使其发动无效并破坏。"},
            )
        raise AssertionError(f"unexpected tool: {tool}")


def action(tool, arguments, rationale=""):
    return json.dumps(
        {"tool": tool, "arguments": arguments, "rationale": rationale},
        ensure_ascii=False,
    )


def request(text="查找我身作盾"):
    return AssistantAgentRequest(text=text, db_path=Path("cards.cdb"))


def test_tool_contracts_expose_all_project_capabilities():
    assert set(ASSISTANT_TOOL_SCHEMAS) == {
        "search_cards",
        "translate_text",
        "adjudicate",
        "finish",
        "rift_search_cards",
        "rift_rules",
        "clarify",
    }


def test_parse_planner_action_validates_and_normalizes_arguments():
    parsed = parse_assistant_planner_action(
        "```json\n"
        + action("search_cards", {"query": "我身作盾"})
        + "\n```"
    )

    assert parsed.tool == "search_cards"
    assert parsed.arguments == {"query": "我身作盾", "top_k": 5}

    with pytest.raises(ValueError, match="未知字段"):
        parse_assistant_planner_action(
            action("search_cards", {"query": "我身作盾", "semantic": True})
        )


def test_agent_calls_search_then_finishes_from_tool_result():
    llm = FakeLlm(
        [
            action("search_cards", {"query": "我身作盾", "top_k": 3}),
            action("finish", {"answer": "找到卡片：我身作盾。"}),
        ]
    )
    runner = FakeToolRunner()

    response = AssistantAgent(llm, runner).run(request())

    assert response.answer == "找到卡片：我身作盾。"
    assert response.exhausted is False
    assert [item.tool for item in response.tool_results] == ["search_cards"]
    assert runner.calls[0][1] == {"query": "我身作盾", "top_k": 3}
    assert "我身作盾" in llm.prompts[1]
    assert response.to_dict()["structured"]["blocks"][0]["tool"] == "search_cards"


def test_agent_can_compose_search_and_translation_tools():
    llm = FakeLlm(
        [
            action("search_cards", {"query": "我身作盾"}),
            action(
                "translate_text",
                {
                    "text": "発動を無効にし破壊する。",
                    "source_lang": "ja",
                    "target_lang": "zh-CN",
                },
            ),
            action("finish", {"answer": "查询和翻译均已完成。"}),
        ]
    )
    runner = FakeToolRunner()

    response = AssistantAgent(llm, runner).run(
        request("查找我身作盾，并翻译这句：発動を無効にし破壊する。")
    )

    assert response.answer == "查询和翻译均已完成。"
    assert [call[0] for call in runner.calls] == ["search_cards", "translate_text"]
    assert [item.tool for item in response.tool_results] == [
        "search_cards",
        "translate_text",
    ]


def test_agent_rejects_duplicate_call_and_uses_bounded_fallback():
    duplicate = action("search_cards", {"query": "我身作盾"})
    llm = FakeLlm([duplicate, duplicate])
    runner = FakeToolRunner()

    response = AssistantAgent(llm, runner).run(
        AssistantAgentRequest(
            text="查找我身作盾",
            db_path=Path("cards.cdb"),
            max_steps=2,
        )
    )

    assert len(runner.calls) == 1
    assert response.exhausted is True
    assert response.answer.startswith("查卡结果")
    assert response.tool_results[1].error == "duplicate tool call"


def test_agent_emits_progress_without_exposing_hidden_reasoning():
    events = []
    llm = FakeLlm(
        [
            action("search_cards", {"query": "我身作盾"}, "需要先查卡"),
            action("finish", {"answer": "完成。"}),
        ]
    )

    AssistantAgent(
        llm,
        FakeToolRunner(),
        progress_callback=events.append,
    ).run(request())

    assert [event["event"] for event in events] == ["planner", "tool", "planner", "final"]
    assert events[0]["message"] == "Agent 选择工具：search_cards"


def test_planner_prompt_is_chinese_and_includes_strict_tool_guidance():
    prompt = build_assistant_planner_prompt("翻译 hello", [])

    assert "统一助手" in prompt
    assert "翻译请求使用 translate_text" in prompt
    assert "只返回一个 JSON object" in prompt


def test_project_tool_runner_adapts_existing_services(monkeypatch, tmp_path):
    settings = Settings.from_env(
        {
            "RAG_DATA_DIR": str(tmp_path),
            "DEEPSEEK_API_KEY": "test-key",
        }
    )
    seen = {}

    def fake_query(query_request, actual_settings):
        seen["query"] = query_request
        assert actual_settings is settings
        return QueryResponse(
            answer="候选：我身作盾",
            results=[
                {
                    "card_id": 1,
                    "name": "我身作盾",
                    "score": 1.0,
                    "source_text": "效果",
                    "reason": "名称匹配",
                }
            ],
            warnings=[],
        )

    def fake_translation(translation_request, actual_settings):
        seen["translation"] = translation_request
        assert actual_settings is settings
        return build_translation_response(
            translation="你好",
            source_lang=translation_request.source_lang,
            target_lang=translation_request.target_lang,
        )

    def fake_adjudication(adjudication_request, actual_settings):
        seen["adjudication"] = adjudication_request
        assert actual_settings is settings
        return AdjudicationResponse(
            scenario=ParsedScenario(original_text=adjudication_request.scenario),
            resolved_cards=(),
            evidence=(),
            reasoning=(),
            answer="裁定原文",
            confidence="low",
            warnings=(),
            simulation=SimulationStatus(status="not_available"),
        )

    monkeypatch.setattr("rag_agent.assistant_agent.execute_query", fake_query)
    monkeypatch.setattr(
        "rag_agent.assistant_agent.execute_translation",
        fake_translation,
    )
    monkeypatch.setattr(
        "rag_agent.assistant_agent.execute_adjudication",
        fake_adjudication,
    )
    runner = ProjectAssistantToolRunner(settings)
    agent_request = AssistantAgentRequest(
        text="组合任务",
        db_path=tmp_path / "cards.cdb",
        semantic=True,
        rerank=True,
        rerank_provider="llm",
        include_specialist_trace=True,
    )

    search = runner.run(
        "search_cards",
        {"query": "我身作盾", "top_k": 3},
        agent_request,
    )
    translation = runner.run(
        "translate_text",
        {"text": "hello", "source_lang": "en", "target_lang": "zh-CN"},
        agent_request,
    )
    adjudication = runner.run(
        "adjudicate",
        {"scenario": "这张卡可以发动吗？"},
        agent_request,
    )

    assert search.data["results"][0]["name"] == "我身作盾"
    assert seen["query"].semantic is True
    assert seen["query"].rerank_provider == "llm"
    assert seen["query"].use_llm is False
    assert translation.data["translation"] == "你好"
    assert seen["adjudication"].use_agent is True
    assert seen["adjudication"].use_llm is True
    assert seen["adjudication"].include_trace is True
    assert adjudication.data["natural_answer"] == "裁定原文"


def test_rift_catalogue_action_supports_filters_and_pages():
    from rag_agent.assistant_agent import parse_assistant_planner_action,build_assistant_planner_prompt
    import json
    action=parse_assistant_planner_action(json.dumps({"tool":"rift_search_cards","arguments":{"filters":{"colors":["红色"],"might":{"min":6},"types":["单位"]},"offset":10,"top_k":10}}))
    assert action.arguments["filters"]["might"]["min"]==6
    assert action.arguments["offset"]==10
    assert "纯红" in build_assistant_planner_prompt("问题",[])


def test_rift_catalogue_rejects_unknown_fields():
    from rag_agent.assistant_agent import parse_assistant_planner_action
    import json,pytest
    with pytest.raises(ValueError):
        parse_assistant_planner_action(json.dumps({"tool":"rift_search_cards","arguments":{"filters":{"attack":6}}}))
