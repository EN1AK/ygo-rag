import json

import pytest

from rag_agent import group_chat
from rag_agent.config import Settings


def payload(**overrides):
    data = dict(group_id=819151070, reply_target_id="m1", messages=[
        dict(id="m1", speaker="成员1", text="这次抽了整整八十发又歪了"),
        dict(id="m2", speaker="机器人", role="assistant", text="晚上好"),
        dict(id="m3", speaker="成员2", text="等下吃什么"),
    ], meme_categories=[dict(category="sad", description="难过")])
    data.update(overrides)
    return data


@pytest.mark.parametrize("extra", [
    dict(reply_target_id="m9"), dict(reply_target_id="m2"), dict(reply_target="旧文本"),
])
def test_bad_or_ambiguous_target_rejected(extra):
    with pytest.raises(ValueError):
        group_chat.parse_group_chat_request(payload(**extra))


@pytest.mark.parametrize("reply,category,expected", [
    ("这次抽了整整八十发，又歪了！", "sad", ("", None)),
    ("这次抽了整整七十发又歪了", "sad", ("", None)),
    ("歪到谁了？", None, ("歪到谁了？", None)),
    ("", None, ("", None)),
    ("", "sad", ("", "sad")),
])
def test_prompt_targets_by_id_and_drops_obvious_echo(monkeypatch, reply, category, expected):
    def invoke(_factory, _settings, _route, messages, **_kwargs):
        user = json.loads(messages[1]["content"])
        assert user["reply_target_id"] == "m1"
        assert "reply_target" not in user
        assert messages[1]["content"].count(payload()["messages"][0]["text"]) == 1
        return json.dumps(dict(reply=reply, meme_category=category))

    monkeypatch.setattr(group_chat, "invoke_group_model", invoke)
    request = group_chat.parse_group_chat_request(payload())
    result = group_chat.execute_group_chat(request, Settings.from_env({}))
    assert (result["reply"], result["meme_category"]) == expected
