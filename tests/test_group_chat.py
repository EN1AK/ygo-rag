import asyncio
import json
from types import SimpleNamespace

import pytest
from rag_agent import group_chat as group
from rag_agent.config import Settings
from rag_agent.web import create_app
from test_web import call_asgi_app


def payload():
    return {"group_id":819151070,"messages":[{"speaker":"成员1","text":"刚刚又抽到废牌了"}],"style_samples":["这下真寄了"]}


def test_group_and_speaker_boundaries():
    data = payload(); data["group_id"] = 123
    with pytest.raises(ValueError):
        group.parse_group_chat_request(data)
    data = payload(); data["messages"][0]["speaker"] = "123456789"
    with pytest.raises(ValueError):
        group.parse_group_chat_request(data)
    data = payload(); data["api_key"] = "secret"
    with pytest.raises(ValueError):
        group.parse_group_chat_request(data)


def test_prompt_keeps_samples_as_data_and_model_has_no_tools(monkeypatch):
    captured = {}
    class Model:
        def bind(self, **kwargs):
            captured["options"] = kwargs
            return self
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content='{"reply":"这下真寄了"}')
    monkeypatch.setattr(group, "create_deepseek_chat_model", lambda settings: Model())
    result = group.execute_group_chat(group.parse_group_chat_request(payload()), Settings.from_env({}))
    assert result == {"reply":"这下真寄了"}
    assert captured["messages"][0]["content"] == group.build_group_prompt(group.parse_group_chat_request(payload()))
    assert json.loads(captured["messages"][1]["content"])["historical_style_examples"] == ["这下真寄了"]
    assert "tools" not in captured["options"]


@pytest.mark.parametrize("reply", ["", "@某人 你好", "[CQ:at,qq=123456789]", "https://example.com", "账号123456789"])
def test_can_abstain_and_blocks_mentions_links_or_identifiers(monkeypatch, reply):
    model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps({"reply":reply})))
    monkeypatch.setattr(group, "create_deepseek_chat_model", lambda settings: SimpleNamespace(bind=lambda **kwargs:model))
    assert group.execute_group_chat(group.parse_group_chat_request(payload()), Settings.from_env({})) == {"reply":""}


def test_web_routes_and_rejects_other_groups():
    app = create_app(group_chat_handler=lambda request: {"reply":"抽牌靠羁绊"})
    response = asyncio.run(call_asgi_app(app,"POST","/api/group-chat",payload()))
    assert response["status"] == 200
    assert json.loads(response["body"])["reply"] == "抽牌靠羁绊"
    data = payload(); data["group_id"] = 123
    response = asyncio.run(call_asgi_app(app,"POST","/api/group-chat",data))
    assert response["status"] == 400
