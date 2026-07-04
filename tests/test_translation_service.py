import pytest

from rag_agent.config import Settings
from rag_agent.translation_service import (
    TranslationRequest,
    build_structured_translation_response,
    build_translation_prompt,
    execute_translation,
    normalize_translation_request,
    split_translation_text,
)


def test_translation_request_defaults_to_auto_source_and_chinese_target():
    request = normalize_translation_request(TranslationRequest(text="hello"))

    assert request.text == "hello"
    assert request.source_lang == "auto"
    assert request.target_lang == "zh-CN"


def test_translation_request_rejects_empty_text():
    with pytest.raises(ValueError, match="text is required"):
        normalize_translation_request(TranslationRequest(text="  "))


def test_translation_prompt_preserves_default_language_direction():
    prompt = build_translation_prompt(TranslationRequest(text="hello"))

    assert "Source language: auto" in prompt
    assert "Target language: zh-CN" in prompt
    assert "<text>\nhello\n</text>" in prompt
    assert "Do not add explanations" in prompt


def test_translation_prompt_supports_explicit_language_direction():
    prompt = build_translation_prompt(
        TranslationRequest(text="你好", source_lang="zh-CN", target_lang="en")
    )

    assert "Source language: zh-CN" in prompt
    assert "Target language: en" in prompt
    assert "<text>\n你好\n</text>" in prompt


def test_structured_translation_response_creates_bot_blocks():
    structured = build_structured_translation_response(
        "第一段\n第二段",
        source_lang="auto",
        target_lang="zh-CN",
    )

    assert structured["version"] == 1
    assert structured["summary"]["block_count"] == 1
    block = structured["blocks"][0]
    assert block["type"] == "translation"
    assert block["text"] == "第一段\n第二段"
    assert block["fields"]["source_lang"] == "auto"
    assert block["fields"]["target_lang"] == "zh-CN"
    assert block["split"] is False
    assert block["truncated"] is False


def test_structured_translation_response_splits_long_text():
    structured = build_structured_translation_response(
        "abcdefghij",
        source_lang="en",
        target_lang="zh-CN",
        max_block_chars=4,
    )

    assert structured["summary"]["block_count"] == 3
    assert structured["summary"]["split"] is True
    assert [block["text"] for block in structured["blocks"]] == ["abcd", "efgh", "ij"]
    assert all(len(block["text"]) <= 4 for block in structured["blocks"])
    assert all(block["split"] is True for block in structured["blocks"])


def test_split_translation_text_prefers_newline_boundary():
    assert split_translation_text("aaa\nbbb", 4) == ["aaa", "bbb"]


def test_execute_translation_uses_configured_llm(monkeypatch):
    prompts = []

    class FakeLlm:
        def invoke(self, prompt):
            prompts.append(prompt)
            return "你好"

    monkeypatch.setattr(
        "rag_agent.translation_service.create_deepseek_chat_model",
        lambda settings: FakeLlm(),
    )

    response = execute_translation(
        TranslationRequest(text="hello"),
        Settings.from_env({"DEEPSEEK_API_KEY": "test-key"}),
    )

    assert response.translation == "你好"
    assert response.target_lang == "zh-CN"
    assert prompts
    assert "hello" in prompts[0]
