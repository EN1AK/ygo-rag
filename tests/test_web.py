import asyncio
import json

from rag_agent.query_service import QueryResponse
from rag_agent.web import create_app, render_index_html


def test_render_index_html_contains_query_form():
    html = render_index_html()

    assert "YGO RAG" in html
    assert "api/query" in html
    assert "semantic" in html
    assert "rerank" in html
    assert "llm_rerank" in html
    assert "llm" in html


def test_web_api_returns_structured_query_response():
    def fake_query_handler(request):
        assert request.query == "有没有效果类似“我身作盾”的卡"
        assert request.semantic is True
        assert request.rerank_provider == "llm"
        return QueryResponse(
            answer="候选：我身作盾",
            results=[
                {
                    "card_id": 1,
                    "name": "我身作盾",
                    "score": 1.0,
                    "source_text": "支付1500基本分，使破坏效果无效并破坏。",
                    "reason": "测试候选",
                }
            ],
            warnings=["测试 warning"],
        )

    app = create_app(query_handler=fake_query_handler)
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/query",
            {
                "query": "有没有效果类似“我身作盾”的卡",
                "semantic": True,
                "rerank": False,
                "llm_rerank": True,
                "llm": False,
                "top_k": 10,
                "rerank_candidates": 20,
            },
        )
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["answer"] == "候选：我身作盾"
    assert payload["results"][0]["name"] == "我身作盾"
    assert payload["warnings"] == ["测试 warning"]


async def call_asgi_app(app, method, path, payload=None):
    body = json.dumps(payload or {}).encode("utf-8")
    messages = []

    async def receive():
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )

    response = {"status": None, "headers": [], "body": b""}
    for message in messages:
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")
    return response
