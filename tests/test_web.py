import asyncio
import json

from rag_agent.query_service import QueryResponse
from rag_agent.adjudication import (
    AdjudicationResponse,
    ParsedScenario,
    SimulationStatus,
)
from rag_agent.translation_service import (
    TranslationResponse,
    build_structured_translation_response,
)
from rag_agent.assistant_agent import (
    AssistantAgentResponse,
    AssistantToolObservation,
)
from rag_agent.web import (
    create_app,
    parse_assistant_request,
    query_response_to_dict,
    render_index_html,
)


def test_render_index_html_contains_query_form():
    html = render_index_html()

    assert "YGO RAG" in html
    assert "api/query" in html
    assert "semantic" in html
    assert "rerank" in html
    assert "llm_rerank" in html
    assert "llm" in html
    assert "api/adjudicate" in html
    assert "adjudication_agent" in html
    assert "adjudication_trace" in html
    assert "renderAgentTrace" in html
    assert "api/translate" not in html


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
                "structured_max_block_chars": 80,
            },
        )
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["answer"] == "候选：我身作盾"
    assert payload["results"][0]["name"] == "我身作盾"
    assert payload["warnings"] == ["测试 warning"]
    assert payload["structured"]["summary"]["result_count"] == 1
    assert payload["structured"]["blocks"][0]["type"] == "card"
    assert len(payload["structured"]["blocks"][0]["text"]) <= 80


def test_query_response_to_dict_includes_structured_filter_diagnostics():
    response = QueryResponse(
        answer="ok",
        results=[],
        warnings=[],
        structured_query={
            "has_filters": True,
            "effect_query": "除外对手墓地",
            "filters": {"card_kind": "monster", "monster_types": ["xyz"], "rank": 4},
        },
        filter_diagnostics={
            "applied": True,
            "total_candidates": 2,
            "filtered_candidates": 1,
            "warnings": [],
        },
    )

    payload = query_response_to_dict(response)

    assert payload["structured_query"]["filters"]["rank"] == 4
    assert payload["filter_diagnostics"]["filtered_candidates"] == 1
    assert payload["structured"]["structured_query"]["filters"]["rank"] == 4
    assert payload["structured"]["filter_diagnostics"]["filtered_candidates"] == 1


def test_web_api_returns_translation_response():
    def fake_translation_handler(request):
        assert request.text == "hello"
        assert request.source_lang == "en"
        assert request.target_lang == "zh-CN"
        assert request.structured_max_block_chars == 4
        return TranslationResponse(
            translation="你好世界",
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            warnings=[],
            structured=build_structured_translation_response(
                "你好世界",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                max_block_chars=request.structured_max_block_chars,
            ),
        )

    app = create_app(translation_handler=fake_translation_handler)
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/translate",
            {
                "text": "hello",
                "source_lang": "en",
                "target_lang": "zh-CN",
                "structured_max_block_chars": 4,
            },
        )
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["translation"] == "你好世界"
    assert payload["source_lang"] == "en"
    assert payload["target_lang"] == "zh-CN"
    assert payload["warnings"] == []
    assert payload["structured"]["blocks"][0]["type"] == "translation"
    assert len(payload["structured"]["blocks"][0]["text"]) <= 4


def test_web_agent_api_returns_unified_structured_response():
    def fake_assistant_handler(request):
        assert request.text == "查找我身作盾并翻译效果"
        assert request.semantic is True
        assert request.rerank_provider == "llm"
        assert request.max_steps == 4
        return AssistantAgentResponse(
            answer="已查卡并翻译。",
            tool_results=(
                AssistantToolObservation(
                    tool="search_cards",
                    ok=True,
                    summary="查卡完成。",
                    data={"results": [{"name": "我身作盾"}]},
                ),
            ),
            warnings=(),
            trace=(),
        )

    app = create_app(assistant_handler=fake_assistant_handler)
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/agent",
            {
                "text": "查找我身作盾并翻译效果",
                "semantic": True,
                "llm_rerank": True,
                "max_steps": 4,
            },
        )
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["answer"] == "已查卡并翻译。"
    assert payload["structured"]["summary"]["tool_call_count"] == 1
    assert payload["structured"]["blocks"][0]["tool"] == "search_cards"
    assert "trace" not in payload


def test_parse_web_agent_request_rejects_credentials_and_bad_steps():
    try:
        parse_assistant_request({"text": "hello", "api_key": "secret"})
    except ValueError as exc:
        assert "server environment" in str(exc)
    else:
        raise AssertionError("credentials should be rejected")

    try:
        parse_assistant_request({"text": "hello", "max_steps": 0})
    except ValueError as exc:
        assert "max_steps must be between 1 and 20" in str(exc)
    else:
        raise AssertionError("invalid max_steps should be rejected")


def test_web_translation_api_rejects_empty_text():
    app = create_app()
    response = asyncio.run(
        call_asgi_app(app, "POST", "/api/translate", {"text": " "})
    )

    assert response["status"] == 400
    payload = json.loads(response["body"].decode("utf-8"))
    assert "text is required" in payload["error"]


def test_web_translation_api_rejects_invalid_block_length():
    app = create_app()
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/translate",
            {"text": "hello", "structured_max_block_chars": 0},
        )
    )

    assert response["status"] == 400
    payload = json.loads(response["body"].decode("utf-8"))
    assert "structured_max_block_chars must be between 1 and 4000" in payload["error"]


def test_web_translation_api_rejects_request_credentials():
    app = create_app()
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/translate",
            {"text": "hello", "api_key": "secret"},
        )
    )

    assert response["status"] == 400
    payload = json.loads(response["body"].decode("utf-8"))
    assert "server environment" in payload["error"]


def test_web_adjudication_api_returns_response():
    def fake_adjudication_handler(request):
        assert request.scenario == "Can Alpha apply?"
        assert request.simulate is True
        assert request.use_agent is True
        assert request.include_trace is True
        return AdjudicationResponse(
            scenario=ParsedScenario(original_text=request.scenario),
            resolved_cards=(),
            evidence=(),
            reasoning=(),
            answer="Answer",
            confidence="low",
            warnings=["missing evidence"],
            simulation=SimulationStatus(status="not_available"),
        )

    app = create_app(adjudication_handler=fake_adjudication_handler)
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/adjudicate",
            {
                "scenario": "Can Alpha apply?",
                "simulate": True,
                "agent": True,
                "trace": True,
            },
        )
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["answer"] == "Answer"
    assert payload["warnings"] == ["missing evidence"]
    assert payload["simulation"]["status"] == "not_available"


def test_web_adjudication_api_rejects_empty_scenario():
    app = create_app()
    response = asyncio.run(
        call_asgi_app(app, "POST", "/api/adjudicate", {"scenario": " "})
    )

    assert response["status"] == 400
    payload = json.loads(response["body"].decode("utf-8"))
    assert "scenario text is required" in payload["error"]


def test_rift_route_returns_pass_through_response():
    def fake_rift_handler(request):
        assert request.query == "反制堆叠有上限吗"
        assert request.top_k == 6
        assert request.mode is None
        assert request.trace is None
        return {
            "answer": "连锁有上限 [R-CR-716.1]",
            "warnings": [],
            "sources": [{"rule_id": "R-CR-716.1", "topic": "连锁"}],
            "mode": "agent",
            "exhausted": False,
            "coverage": {"unresolved_mentions": [], "ambiguous_mentions": []},
        }

    app = create_app(rift_handler=fake_rift_handler)
    response = asyncio.run(
        call_asgi_app(
            app, "POST", "/api/rift/query", {"query": "反制堆叠有上限吗"}
        )
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["answer"] == "连锁有上限 [R-CR-716.1]"
    assert payload["mode"] == "agent"
    assert payload["coverage"] == {
        "unresolved_mentions": [],
        "ambiguous_mentions": [],
    }


def test_rift_route_forwards_explicit_options_to_handler():
    seen = {}

    def fake_rift_handler(request):
        seen["mode"] = request.mode
        seen["trace"] = request.trace
        seen["top_k"] = request.top_k
        return {"answer": "a", "warnings": [], "sources": []}

    app = create_app(rift_handler=fake_rift_handler)
    response = asyncio.run(
        call_asgi_app(
            app,
            "POST",
            "/api/rift/query",
            {"query": "q", "top_k": 9, "mode": "oneshot", "trace": True},
        )
    )

    assert response["status"] == 200
    assert seen == {"mode": "oneshot", "trace": True, "top_k": 9}


def test_rift_route_rejects_invalid_body_with_400():
    def fake_rift_handler(request):
        raise AssertionError("handler must not run on validation failure")

    app = create_app(rift_handler=fake_rift_handler)
    response = asyncio.run(
        call_asgi_app(app, "POST", "/api/rift/query", {"query": " "})
    )

    assert response["status"] == 400
    payload = json.loads(response["body"].decode("utf-8"))
    assert "query" in payload["error"]


def test_rift_route_maps_gateway_error_categories():
    from rag_agent.rift_gateway import RiftGatewayError

    cases = [
        ("unavailable", 502),
        ("timeout", 504),
        ("upstream", 502),
    ]
    for category, expected_status in cases:
        def fake_rift_handler(request, category=category):
            raise RiftGatewayError(category, f"boom {category}")

        app = create_app(rift_handler=fake_rift_handler)
        response = asyncio.run(
            call_asgi_app(app, "POST", "/api/rift/query", {"query": "q"})
        )

        assert response["status"] == expected_status, category
        payload = json.loads(response["body"].decode("utf-8"))
        assert payload["error"] == f"boom {category}"


def test_rift_alias_route_for_kami_man_base_composition():
    # kami-man 以 {base}/api/query 拼接：base=http://127.0.0.1:7860/api/rift
    # 时拼出 /api/rift/api/query，应与规整路由同 handler。
    app = create_app(
        rift_handler=lambda request: {"answer": "a", "warnings": [], "sources": []}
    )
    response = asyncio.run(
        call_asgi_app(app, "POST", "/api/rift/api/query", {"query": "q"})
    )

    assert response["status"] == 200
    payload = json.loads(response["body"].decode("utf-8"))
    assert payload["answer"] == "a"


def test_unknown_rift_path_still_404():
    app = create_app(rift_handler=lambda request: {})
    response = asyncio.run(
        call_asgi_app(app, "POST", "/api/rift/anything", {"query": "q"})
    )

    assert response["status"] == 404


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
