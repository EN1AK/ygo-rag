import asyncio
import json
from types import SimpleNamespace

import pytest

from rag_agent import card_translation as card
from rag_agent.config import Settings
from rag_agent.translation_service import build_translation_response
from rag_agent.web import create_app
from test_web import call_asgi_app


@pytest.mark.parametrize("images", [None, [], ["https://example.com/a", "https://example.com/b"],
    ["file:///secret"], ["http://127.0.0.1/a"], ["http://[::1]/a"], ["http://localhost/a"], [123]])
def test_reject_invalid_images(images):
    with pytest.raises(ValueError):
        card.parse_card_translation_request({"images": images})


def test_image_request_preserves_signed_url_and_forbids_credentials():
    url = "https://multimedia.nt.qq.com.cn/download?appid=1&rkey=abc%2Bdef"
    assert card.parse_card_translation_request({"images": [url]}).image_url == url
    with pytest.raises(ValueError, match="credentials"):
        card.parse_card_translation_request({"images": [url], "api_key": "secret"})


@pytest.mark.parametrize("url", ["data:image/jpeg;base64,", "data:image/jpeg;base64,@@@", "data:text/html;base64,YQ=="])
def test_reject_invalid_inline_images(url):
    with pytest.raises(ValueError):
        card.parse_card_translation_request({"images": [url]})


def test_inline_image_is_accepted():
    url = "data:image/jpeg;base64,/9j/"
    assert card.parse_card_translation_request({"images": [url]}).image_url == url


def test_vision_payload_model_and_pendulum_output(monkeypatch):
    captured = {}
    class FakeModel:
        def bind(self, **kwargs):
            captured["binding"] = kwargs
            return self
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content=json.dumps({"readable": True, "cn_name": "测试卡", "effect": "①：测试效果。", "pendulum_effect": "每回合一次。"}))
    def factory(settings):
        captured["model"] = settings.deepseek_model
        return FakeModel()
    monkeypatch.setattr(card, "create_deepseek_chat_model", factory)
    monkeypatch.setenv("DEEPSEEK_VISION_MODEL", "vision-model")
    settings = Settings.from_env({"DEEPSEEK_MODEL": "text-model"})
    result = card.execute_card_translation(card.CardTranslationRequest("https://example.com/card.jpg"), settings)
    assert captured["model"] == "vision-model" and settings.deepseek_model == "text-model"
    assert captured["messages"][1]["content"][1]["image_url"]["url"] == "https://example.com/card.jpg"
    assert "中文卡名：测试卡" in result.translation
    assert "效果：\n①：测试效果。" in result.translation and "灵摆效果：\n每回合一次。" in result.translation
    assert "机器翻译" in result.translation


@pytest.mark.parametrize("content,error", [
    ('{"readable":false}', card.UnreadableCardError),
    ('{"readable":"true"}', RuntimeError),
    ('{"readable":true,"cn_name":"卡","effect":"","pendulum_effect":""}', RuntimeError),
    ('not json', RuntimeError),
])
def test_unreadable_and_malformed_outputs_are_failures(monkeypatch, content, error):
    fake = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=content))
    monkeypatch.setattr(card, "create_deepseek_chat_model", lambda settings: SimpleNamespace(bind=lambda **kwargs: fake))
    with pytest.raises(error):
        card.execute_card_translation(card.CardTranslationRequest("https://example.com/card.jpg"), Settings.from_env({}))


def test_endpoint_routes_image_and_reports_unreadable():
    def handler(request):
        assert request.image_url == "https://example.com/card.jpg"
        return build_translation_response(translation="中文卡名：测试\n效果：测试")
    app = create_app(card_translation_handler=handler)
    result = asyncio.run(call_asgi_app(app, "POST", "/api/card-translate", {"images": ["https://example.com/card.jpg"]}))
    assert result["status"] == 200
    assert "中文卡名" in json.loads(result["body"])["translation"]
    def unreadable(request):
        raise card.UnreadableCardError("unreadable")
    app = create_app(card_translation_handler=unreadable)
    result = asyncio.run(call_asgi_app(app, "POST", "/api/card-translate", {"images": ["https://example.com/card.jpg"]}))
    assert result["status"] == 422
