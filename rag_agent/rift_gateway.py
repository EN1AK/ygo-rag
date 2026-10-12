"""Riftbound 规则 RAG 网关/适配层。

本机统一入口（rag_agent/web.py 的 POST /api/rift/query）经本层代理到 riftsim
的 rift-rag 服务（默认 http://127.0.0.1:7862）。职责：

- 请求模型校验（query/top_k/mode/trace），失败抛
  RiftGatewayError(category="validation")；
- transport 默认基于 requests，可注入（测试/其他实现），transport 自身把
  连接失败/超时/非 200 翻译成 RiftGatewayError(unavailable/timeout/upstream)；
- 成功响应校验 legacy 核心形状（answer/warnings/sources），畸形抛 upstream；
  v2 字段（resolved_cards/coverage/exhausted/mode/trace）仅在上游存在时透传，
  绝不合成。

与 riftsim 契约一致：mode/trace 仅在调用方显式提供时才发给上游，上游默认
agent 模式；错误 JSON 一律 {"error": str}。
"""
from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from rag_agent.config import Settings

VALID_MODES = ("agent", "oneshot")
DEFAULT_TOP_K = 6
MAX_TOP_K = 20

CATEGORY_VALIDATION = "validation"
CATEGORY_UNAVAILABLE = "unavailable"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_UPSTREAM = "upstream"

# 透传的契约 v2 字段（仅当上游响应存在时复制）
V2_PASSTHROUGH_KEYS = ("resolved_cards", "coverage", "exhausted", "mode", "trace")

Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


class RiftGatewayError(Exception):
    """按类别区分的网关错误；category ∈ validation/unavailable/timeout/upstream。"""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class RiftQueryRequest:
    """网关请求模型；构造即校验（CLI 与 web 路由共用同一份约束）。"""

    query: str
    top_k: int = DEFAULT_TOP_K
    mode: str | None = None
    trace: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise RiftGatewayError(
                CATEGORY_VALIDATION, "query must be a non-empty string."
            )
        object.__setattr__(self, "query", self.query.strip())
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise RiftGatewayError(CATEGORY_VALIDATION, "top_k must be an integer.")
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise RiftGatewayError(
                CATEGORY_VALIDATION, f"top_k must be between 1 and {MAX_TOP_K}."
            )
        if self.mode is not None and self.mode not in VALID_MODES:
            raise RiftGatewayError(
                CATEGORY_VALIDATION, "mode must be 'agent' or 'oneshot'."
            )
        if self.trace is not None and not isinstance(self.trace, bool):
            raise RiftGatewayError(CATEGORY_VALIDATION, "trace must be a boolean.")


def parse_rift_gateway_request(payload: Any) -> RiftQueryRequest:
    """把 JSON 请求体解析为 RiftQueryRequest；非法输入抛 validation 错误。"""
    if not isinstance(payload, Mapping):
        raise RiftGatewayError(
            CATEGORY_VALIDATION, "request body must be a JSON object."
        )
    return RiftQueryRequest(
        query=payload.get("query"),
        top_k=payload.get("top_k", DEFAULT_TOP_K),
        mode=payload.get("mode"),
        trace=payload.get("trace"),
    )


def request_to_upstream_body(request: RiftQueryRequest) -> dict[str, Any]:
    """组装发往上游的 JSON 体；mode/trace 仅在显式提供时携带。"""
    body: dict[str, Any] = {"query": request.query, "top_k": request.top_k}
    if request.mode is not None:
        body["mode"] = request.mode
    if request.trace is not None:
        body["trace"] = request.trace
    return body


def _default_transport(
    url: str, payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    import requests

    try:
        response = requests.post(url, json=dict(payload), timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise RiftGatewayError(
            CATEGORY_TIMEOUT, f"upstream timeout after {timeout:g}s."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RiftGatewayError(
            CATEGORY_UNAVAILABLE, f"upstream unavailable: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RiftGatewayError(
            CATEGORY_UNAVAILABLE, f"upstream request failed: {exc}"
        ) from exc

    if response.status_code != 200:
        detail = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            detail = body["error"]
        message = f"upstream returned HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail}"
        raise RiftGatewayError(CATEGORY_UPSTREAM, message)

    try:
        parsed = response.json()
    except ValueError as exc:
        raise RiftGatewayError(
            CATEGORY_UPSTREAM, "malformed success response: not JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise RiftGatewayError(
            CATEGORY_UPSTREAM, "malformed success response: not a JSON object."
        )
    return parsed


def execute_rift_query(
    request: RiftQueryRequest,
    settings: Settings | None = None,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """POST {base}/api/query 并组装透传响应；失败抛 RiftGatewayError。"""
    if transport is None and os.getenv("RIFTBOUND_RAG_TRANSPORT","library")=="library":
        from rag_agent.rift_tools import query_rules
        try:
            return assemble_rift_gateway_response(query_rules(request.query,request.top_k,request.trace or False,request.mode))
        except TimeoutError as exc:raise RiftGatewayError(CATEGORY_TIMEOUT,"符文查询超时") from exc
        except OSError as exc:raise RiftGatewayError(CATEGORY_UNAVAILABLE,"符文知识库不可用") from exc
    settings = settings or Settings.from_env()
    transport = transport or _default_transport
    url = f"{settings.riftbound_rag_base_url.rstrip('/')}/api/query"
    raw = transport(
        url, request_to_upstream_body(request),
        settings.riftbound_rag_timeout_seconds,
    )
    return assemble_rift_gateway_response(raw)


def assemble_rift_gateway_response(raw: Any) -> dict[str, Any]:
    """校验上游成功响应的 legacy 核心形状并透传 v2 字段。"""
    if not isinstance(raw, Mapping):
        raise RiftGatewayError(
            CATEGORY_UPSTREAM, "malformed success response: not a JSON object."
        )
    answer = raw.get("answer")
    warnings = raw.get("warnings")
    sources = raw.get("sources")
    if not isinstance(answer, str):
        raise RiftGatewayError(
            CATEGORY_UPSTREAM, "malformed success response: answer must be a string."
        )
    if not isinstance(warnings, list):
        raise RiftGatewayError(
            CATEGORY_UPSTREAM,
            "malformed success response: warnings must be a list.",
        )
    if not isinstance(sources, list):
        raise RiftGatewayError(
            CATEGORY_UPSTREAM, "malformed success response: sources must be a list."
        )
    for index, source in enumerate(sources):
        if (
            not isinstance(source, Mapping)
            or not isinstance(source.get("rule_id"), str)
            or not isinstance(source.get("topic"), str)
        ):
            raise RiftGatewayError(
                CATEGORY_UPSTREAM,
                f"malformed success response: sources[{index}] must contain "
                "string rule_id and topic.",
            )
    response: dict[str, Any] = {
        "answer": answer,
        "warnings": warnings,
        "sources": sources,
    }
    for key in V2_PASSTHROUGH_KEYS:
        if key in raw:
            response[key] = raw[key]
    return response
