import pytest
import requests

from rag_agent.config import Settings
from rag_agent.rift_gateway import (
    CATEGORY_TIMEOUT,
    CATEGORY_UNAVAILABLE,
    CATEGORY_UPSTREAM,
    CATEGORY_VALIDATION,
    RiftGatewayError,
    RiftQueryRequest,
    assemble_rift_gateway_response,
    execute_rift_query,
    parse_rift_gateway_request,
    request_to_upstream_body,
)


def _settings(**overrides):
    env = {
        "RIFTBOUND_RAG_BASE_URL": "http://127.0.0.1:7862",
        "RIFTBOUND_RAG_TIMEOUT_SECONDS": "200",
    }
    env.update(overrides)
    return Settings.from_env(env)


def _request(**kwargs):
    kwargs.setdefault("query", "反制堆叠有上限吗")
    return RiftQueryRequest(**kwargs)


# ---------- 1.1 请求模型与校验 ----------

def test_minimal_request_defaults():
    request = parse_rift_gateway_request({"query": " 什么是连锁 "})

    assert request.query == "什么是连锁"
    assert request.top_k == 6
    assert request.mode is None
    assert request.trace is None


def test_explicit_options_forwarded_verbatim():
    request = parse_rift_gateway_request(
        {"query": "q", "top_k": 12, "mode": "oneshot", "trace": True}
    )

    assert request.top_k == 12
    assert request.mode == "oneshot"
    assert request.trace is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": 123},
        "not-a-dict",
        {"query": "q", "top_k": "6"},
        {"query": "q", "top_k": 6.0},
        {"query": "q", "top_k": True},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": 21},
        {"query": "q", "mode": "turbo"},
        {"query": "q", "trace": "yes"},
    ],
)
def test_invalid_requests_raise_validation_error(payload):
    with pytest.raises(RiftGatewayError) as excinfo:
        parse_rift_gateway_request(payload)

    assert excinfo.value.category == CATEGORY_VALIDATION


def test_validation_failure_never_touches_transport():
    calls = []

    def stub_transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {}

    with pytest.raises(RiftGatewayError):
        _request(query="  ")
    with pytest.raises(RiftGatewayError):
        parse_rift_gateway_request({"query": "q", "mode": "nope"})
    assert calls == []


def test_upstream_body_omits_unset_options():
    body = request_to_upstream_body(_request())

    assert body == {"query": "反制堆叠有上限吗", "top_k": 6}
    assert "mode" not in body and "trace" not in body


def test_upstream_body_includes_explicit_options():
    body = request_to_upstream_body(_request(mode="agent", trace=False))

    assert body["mode"] == "agent" and body["trace"] is False


# ---------- 1.2 上游调用与失败分类 ----------

def test_execute_posts_to_upstream_api_query():
    calls = []

    def stub_transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"answer": "a", "warnings": [], "sources": []}

    result = execute_rift_query(
        _request(top_k=8), settings=_settings(), transport=stub_transport
    )

    assert result["answer"] == "a"
    assert calls == [
        (
            "http://127.0.0.1:7862/api/query",
            {"query": "反制堆叠有上限吗", "top_k": 8},
            200.0,
        )
    ]


@pytest.mark.parametrize(
    "category",
    [CATEGORY_UNAVAILABLE, CATEGORY_TIMEOUT, CATEGORY_UPSTREAM],
)
def test_gateway_errors_from_transport_propagate(category):
    def stub_transport(url, payload, timeout):
        raise RiftGatewayError(category, f"stubbed {category}")

    with pytest.raises(RiftGatewayError) as excinfo:
        execute_rift_query(_request(), settings=_settings(),
                           transport=stub_transport)

    assert excinfo.value.category == category
    assert f"stubbed {category}" in str(excinfo.value)


def test_default_transport_connection_error_is_unavailable(monkeypatch):
    def raise_connection(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", raise_connection)

    with pytest.raises(RiftGatewayError) as excinfo:
        execute_rift_query(_request(), settings=_settings())

    assert excinfo.value.category == CATEGORY_UNAVAILABLE


def test_default_transport_timeout_is_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(requests, "post", raise_timeout)

    with pytest.raises(RiftGatewayError) as excinfo:
        execute_rift_query(_request(), settings=_settings())

    assert excinfo.value.category == CATEGORY_TIMEOUT
    assert "200" in str(excinfo.value)


class _FakeResponse:
    def __init__(self, status_code, body=None, json_error=None):
        self.status_code = status_code
        self._body = body
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


def test_default_transport_non_200_includes_upstream_error_text(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _FakeResponse(400, {"error": "query 必须是非空字符串"}),
    )

    with pytest.raises(RiftGatewayError) as excinfo:
        execute_rift_query(_request(), settings=_settings())

    assert excinfo.value.category == CATEGORY_UPSTREAM
    assert "400" in str(excinfo.value)
    assert "query 必须是非空字符串" in str(excinfo.value)


def test_default_transport_non_200_without_json_body(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _FakeResponse(500, json_error=ValueError("no json")),
    )

    with pytest.raises(RiftGatewayError) as excinfo:
        execute_rift_query(_request(), settings=_settings())

    assert excinfo.value.category == CATEGORY_UPSTREAM
    assert "HTTP 500" in str(excinfo.value)


def test_default_transport_success_body_must_be_json_object(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: _FakeResponse(200, ["not", "a", "dict"])
    )

    with pytest.raises(RiftGatewayError) as excinfo:
        execute_rift_query(_request(), settings=_settings())

    assert excinfo.value.category == CATEGORY_UPSTREAM
    assert "malformed" in str(excinfo.value)


# ---------- 1.3 响应装配与 v2 透传 ----------

def test_legacy_contract_passed_through_exactly():
    raw = {
        "answer": "答案",
        "warnings": ["w1"],
        "sources": [{"rule_id": "R-CR-716.1", "topic": "连锁"}],
    }

    result = assemble_rift_gateway_response(raw)

    assert result == raw
    assert set(result) == {"answer", "warnings", "sources"}


def test_v2_fields_copied_verbatim_when_present():
    raw = {
        "answer": "答案 [R-CR-1]",
        "warnings": [],
        "sources": [{"rule_id": "R-CR-1", "topic": "t"}],
        "resolved_cards": [{"card_id": "OGN-242", "mention": "海兽钓钩"}],
        "coverage": {"unresolved_mentions": ["某卡"],
                     "ambiguous_mentions": [{"mention": "蔚"}]},
        "exhausted": False,
        "mode": "agent",
        "trace": [{"step": 1, "tool": "search_rules"}],
    }

    result = assemble_rift_gateway_response(raw)

    assert result == raw  # v2 字段值与上游完全一致
    assert result["coverage"]["unresolved_mentions"] == ["某卡"]


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-dict",
        {},
        {"answer": 1, "warnings": [], "sources": []},
        {"answer": "a"},
        {"answer": "a", "warnings": "w", "sources": []},
        {"answer": "a", "warnings": [], "sources": "s"},
        {"answer": "a", "warnings": [], "sources": [{"rule_id": 1, "topic": "t"}]},
        {"answer": "a", "warnings": [], "sources": [{"rule_id": "r"}]},
        {"answer": "a", "warnings": [], "sources": ["oops"]},
    ],
)
def test_malformed_success_bodies_raise_upstream_error(raw):
    with pytest.raises(RiftGatewayError) as excinfo:
        assemble_rift_gateway_response(raw)

    assert excinfo.value.category == CATEGORY_UPSTREAM
    assert "malformed success response" in str(excinfo.value)


@pytest.fixture(autouse=True)
def legacy_http_transport(monkeypatch):
    # Existing transport contract cases exercise the explicit rollback mode.
    monkeypatch.setenv("RIFTBOUND_RAG_TRANSPORT","http")

def test_default_transport_calls_library_without_http(monkeypatch):
    monkeypatch.delenv("RIFTBOUND_RAG_TRANSPORT")
    seen=[]
    def query(*args):
        seen.append(args)
        return {"answer":"库调用结果","warnings":[],"sources":[]}
    monkeypatch.setattr("rag_agent.rift_tools.query_rules",query)
    result=execute_rift_query(_request())
    assert result["answer"]=="库调用结果" and len(seen)==1
