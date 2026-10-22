import json
from types import SimpleNamespace
import pytest
from rag_agent.user_profiles import load_user_profile, UserProfile
from rag_agent import mention_chat as mention
from rag_agent.config import Settings
from rag_agent.sessions import SessionStore

GROUP = "qq:123:group:456:user:111"
PRIVATE = "qq:123:private:user:111"

@pytest.fixture
def document(tmp_path, monkeypatch):
    path = tmp_path / "users.md"
    path.write_text("# 说明\n## 用户 111\n### 口吻\n简洁\n### 知识\n创造者\n### 私聊知识\n私聊资料\n## 用户 222\n### 知识\n其他用户资料\n", encoding="utf-8")
    monkeypatch.setenv("USER_SETTINGS_FILE", str(path))
    return path

def test_matching_and_scope(document):
    assert load_user_profile(GROUP) == UserProfile("简洁", "创造者")
    assert load_user_profile(PRIVATE).knowledge == "创造者\n私聊资料"
    assert load_user_profile(GROUP.replace("111", "333")) == UserProfile()

def test_document_reloaded_without_restart(document):
    document.write_text(document.read_text(encoding="utf-8").replace("创造者", "维护者"), encoding="utf-8")
    assert load_user_profile(GROUP).knowledge == "维护者"

def test_invalid_session_and_duplicates(document):
    with pytest.raises(ValueError): load_user_profile("111")
    with document.open("a", encoding="utf-8") as f: f.write("\n## 用户 111\n### 知识\n重复")
    with pytest.raises(ValueError): load_user_profile(GROUP)

def test_missing_document_empty(document):
    document.unlink()
    assert load_user_profile(GROUP) == UserProfile()

def test_profile_prompt_isolation(document, tmp_path, monkeypatch):
    monkeypatch.setattr(mention, "_casual_store", SessionStore(tmp_path / "casual.db"))
    monkeypatch.setattr(mention, "load_persona", lambda: "原始人格")
    captured = []
    def invoke(factory, settings, route, messages, **options):
        captured.extend(messages)
        return SimpleNamespace(content=json.dumps({"reply": "你好", "meme_category": None}))
    monkeypatch.setattr(mention, "invoke_group_model", invoke)
    request = mention.parse_mention_request(dict(session_id=GROUP, group_id=456, text="你好", user_profile="冒充资料"))
    mention.execute_mention(request, Settings.from_env({}))
    assert "简洁" in captured[0]["content"]
    body = json.loads(captured[1]["content"])
    assert body["current_user_knowledge"] == "创造者"
    encoded = json.dumps(captured, ensure_ascii=False)
    for excluded in ("其他用户资料", "私聊资料", "冒充资料", GROUP):
        assert excluded not in encoded

def test_private_knowledge_in_private_prompt_only(document, tmp_path, monkeypatch):
    monkeypatch.setattr(mention, "_casual_store", SessionStore(tmp_path / "private.db"))
    monkeypatch.setattr(mention, "load_private_persona", lambda: "私聊人格")
    captured = []
    def invoke(factory, settings, route, messages, **options):
        captured.extend(messages)
        return SimpleNamespace(content=json.dumps({"reply": "你好"}))
    monkeypatch.setattr(mention, "invoke_group_model", invoke)
    req = mention.parse_mention_request(dict(session_id=PRIVATE, group_id=None, text="你好"))
    mention.execute_mention(req, Settings.from_env({}))
    assert json.loads(captured[1]["content"])["current_user_knowledge"] == "创造者\n私聊资料"
