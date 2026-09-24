from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from rag_agent.config import Settings
from rag_agent.adjudication_service import (
    AdjudicationRequest,
    AdjudicationResponse,
    adjudication_response_to_dict,
    execute_adjudication,
)
from rag_agent.query_service import (
    QueryRequest,
    QueryResponse,
    build_structured_response,
    execute_query,
)
from rag_agent.translation_service import (
    TranslationRequest,
    TranslationResponse,
    execute_translation,
    translation_response_to_dict,
)
from rag_agent.assistant_agent import (
    AssistantAgentRequest,
    AssistantAgentResponse,
    execute_assistant_agent,
)
from rag_agent.rift_gateway import (
    CATEGORY_TIMEOUT,
    CATEGORY_UNAVAILABLE,
    CATEGORY_UPSTREAM,
    CATEGORY_VALIDATION,
    RiftGatewayError,
    RiftQueryRequest,
    execute_rift_query,
    parse_rift_gateway_request,
)

QueryHandler = Callable[[QueryRequest], QueryResponse]
TranslationHandler = Callable[[TranslationRequest], TranslationResponse]
AdjudicationHandler = Callable[[AdjudicationRequest], AdjudicationResponse]
AssistantHandler = Callable[[AssistantAgentRequest], AssistantAgentResponse]
RiftHandler = Callable[[RiftQueryRequest], Mapping[str, Any]]

_RIFT_ERROR_STATUS = {
    CATEGORY_VALIDATION: 400,
    CATEGORY_UNAVAILABLE: 502,
    CATEGORY_TIMEOUT: 504,
    CATEGORY_UPSTREAM: 502,
}


def create_app(
    query_handler: QueryHandler | None = None,
    translation_handler: TranslationHandler | None = None,
    adjudication_handler: AdjudicationHandler | None = None,
    assistant_handler: AssistantHandler | None = None,
    rift_handler: RiftHandler | None = None,
):
    handler = query_handler or _default_query_handler
    translate = translation_handler or _default_translation_handler
    adjudicate = adjudication_handler or _default_adjudication_handler
    run_assistant = assistant_handler or _default_assistant_handler
    run_rift = rift_handler or _default_rift_handler

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            await _handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            await _send_response(send, 404, {"error": "Unsupported scope."})
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "/")
        try:
            if method == "GET" and path == "/":
                await _send_html(send, render_index_html())
                return
            if method == "POST" and path == "/api/query":
                payload = await _read_json_body(receive)
                request = parse_query_request(payload)
                response = handler(request)
                await _send_response(
                    send,
                    200,
                    query_response_to_dict(
                        response,
                        structured_max_block_chars=_optional_int_in_range(
                            payload.get("structured_max_block_chars"),
                            "structured_max_block_chars",
                            minimum=1,
                            maximum=4000,
                        ),
                    ),
                )
                return
            if method == "POST" and path == "/api/translate":
                payload = await _read_json_body(receive)
                request = parse_translation_request(payload)
                response = translate(request)
                await _send_response(send, 200, translation_response_to_dict(response))
                return
            if method == "POST" and path == "/api/adjudicate":
                payload = await _read_json_body(receive)
                request = parse_adjudication_request(payload)
                response = adjudicate(request)
                await _send_response(send, 200, adjudication_response_to_dict(response))
                return
            if method == "POST" and path == "/api/agent":
                payload = await _read_json_body(receive)
                request = parse_assistant_request(payload)
                response = run_assistant(request)
                await _send_response(
                    send,
                    200,
                    response.to_dict(
                        include_trace=_as_bool(payload.get("trace", False))
                    ),
                )
                return
            # /api/rift/query 为规整路由；/api/rift/api/query 为 kami-man
            # base URL 拼接（{base}/api/query）下的消费路径，二者同 handler。
            if method == "POST" and path in ("/api/rift/query", "/api/rift/api/query"):
                payload = await _read_json_body(receive)
                try:
                    response = run_rift(parse_rift_gateway_request(payload))
                except RiftGatewayError as exc:
                    await _send_response(
                        send,
                        _RIFT_ERROR_STATUS.get(exc.category, 500),
                        {"error": str(exc)},
                    )
                    return
                await _send_response(send, 200, response)
                return
            await _send_response(send, 404, {"error": "Not found."})
        except ValueError as exc:
            await _send_response(send, 400, {"error": str(exc)})
        except Exception as exc:
            await _send_response(send, 500, {"error": str(exc)})

    return app


async def _handle_lifespan(receive, send) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


def parse_query_request(payload: Mapping[str, Any]) -> QueryRequest:
    settings = Settings.from_env()
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("query is required.")

    db_path_value = str(payload.get("db_path") or settings.cards_db_path).strip()
    return QueryRequest(
        query=query,
        db_path=Path(db_path_value),
        top_k=_int_in_range(payload.get("top_k", 10), "top_k", minimum=1, maximum=50),
        semantic=_as_bool(payload.get("semantic", False)),
        rerank=_as_bool(payload.get("rerank", False)),
        rerank_provider=_parse_rerank_provider(payload),
        use_llm=_as_bool(payload.get("llm", payload.get("use_llm", False))),
        rerank_candidates=_int_in_range(
            payload.get("rerank_candidates", 20),
            "rerank_candidates",
            minimum=1,
            maximum=200,
        ),
    )


def parse_translation_request(payload: Mapping[str, Any]) -> TranslationRequest:
    forbidden_keys = {
        "api_key",
        "deepseek_api_key",
        "DEEPSEEK_API_KEY",
        "authorization",
    }
    if any(key in payload for key in forbidden_keys):
        raise ValueError("DeepSeek credentials must be configured in the server environment.")

    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("text is required.")
    return TranslationRequest(
        text=text,
        source_lang=str(payload.get("source_lang") or "auto").strip() or "auto",
        target_lang=str(payload.get("target_lang") or "zh-CN").strip() or "zh-CN",
        structured_max_block_chars=_optional_int_in_range(
            payload.get("structured_max_block_chars"),
            "structured_max_block_chars",
            minimum=1,
            maximum=4000,
        ),
    )


def parse_adjudication_request(payload: Mapping[str, Any]) -> AdjudicationRequest:
    settings = Settings.from_env()
    scenario = str(payload.get("scenario", payload.get("query", ""))).strip()
    if not scenario:
        raise ValueError("scenario text is required.")
    return AdjudicationRequest(
        scenario=scenario,
        db_path=Path(str(payload.get("db_path") or settings.cards_db_path).strip()),
        ygocdb_cache_path=_optional_path(payload.get("ygocdb_cache_path")),
        rules_epub_path=_optional_path(payload.get("rules_epub_path")),
        use_llm=_as_bool(payload.get("llm", payload.get("use_llm", False))),
        simulate=_as_bool(payload.get("simulate", False)),
        allow_ygocdb_fetch=_as_bool(payload.get("allow_ygocdb_fetch", False)),
        use_agent=_as_bool(payload.get("agent", payload.get("use_agent", False))),
        include_trace=_as_bool(payload.get("trace", payload.get("include_trace", False))),
    )


def parse_assistant_request(payload: Mapping[str, Any]) -> AssistantAgentRequest:
    forbidden_keys = {
        "api_key",
        "deepseek_api_key",
        "DEEPSEEK_API_KEY",
        "authorization",
    }
    if any(key in payload for key in forbidden_keys):
        raise ValueError("DeepSeek credentials must be configured in the server environment.")
    settings = Settings.from_env()
    text = str(payload.get("text", payload.get("query", ""))).strip()
    if not text:
        raise ValueError("text is required.")
    rerank_provider = _parse_rerank_provider(payload)
    return AssistantAgentRequest(
        text=text,
        db_path=Path(str(payload.get("db_path") or settings.cards_db_path).strip()),
        semantic=_as_bool(payload.get("semantic", False)),
        rerank=_as_bool(payload.get("rerank", False)),
        rerank_provider=rerank_provider,
        max_steps=_int_in_range(
            payload.get("max_steps", 6),
            "max_steps",
            minimum=1,
            maximum=20,
        ),
        ygocdb_cache_path=_optional_path(payload.get("ygocdb_cache_path")),
        rules_epub_path=_optional_path(payload.get("rules_epub_path")),
        include_specialist_trace=_as_bool(payload.get("specialist_trace", False)),
        structured_max_block_chars=_optional_int_in_range(
            payload.get("structured_max_block_chars"),
            "structured_max_block_chars",
            minimum=1,
            maximum=4000,
        ),
    )


def query_response_to_dict(
    response: QueryResponse,
    *,
    structured_max_block_chars: int | None = None,
) -> dict[str, Any]:
    return {
        "answer": response.answer,
        "results": response.results,
        "warnings": response.warnings,
        "structured_query": response.structured_query,
        "filter_diagnostics": response.filter_diagnostics,
        "structured": build_structured_response(
            response,
            max_block_chars=structured_max_block_chars,
        ),
    }


def render_index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YGO RAG</title>
  <style>
    :root { color-scheme: light dark; font-family: "Segoe UI", system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; }
    main { max-width: 1080px; margin: 0 auto; padding: 32px 20px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { color: #9ca3af; }
    textarea, input { width: 100%; box-sizing: border-box; border: 1px solid #374151; border-radius: 10px; background: #0b1220; color: #f9fafb; padding: 12px; }
    textarea { min-height: 96px; resize: vertical; font-size: 16px; }
    label { display: block; margin: 14px 0 6px; color: #d1d5db; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
    .checks { display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0; }
    .checks label { display: flex; align-items: center; gap: 8px; margin: 0; }
    .checks input { width: auto; }
    button { border: 0; border-radius: 10px; padding: 12px 18px; background: #2563eb; color: white; font-size: 16px; cursor: pointer; }
    button:disabled { background: #4b5563; cursor: wait; }
    .panel { margin-top: 22px; border: 1px solid #374151; border-radius: 14px; background: #030712; padding: 18px; }
    .answer { white-space: pre-wrap; line-height: 1.65; }
    .warning { color: #fbbf24; }
    .card { border-top: 1px solid #1f2937; padding-top: 14px; margin-top: 14px; }
    .card-title { color: #93c5fd; font-weight: 700; }
    .muted { color: #9ca3af; }
    code { background: #111827; padding: 2px 5px; border-radius: 5px; }
  </style>
</head>
<body>
<main>
  <h1>YGO RAG</h1>
  <p>本地游戏王卡片效果相似检索。DeepSeek API Key、模型路径和 Chroma 路径仍从环境变量读取。</p>
  <form id="query-form">
    <label for="query">问题</label>
    <textarea id="query" name="query" required>有没有效果类似“我身作盾”的卡</textarea>
    <div class="grid">
      <div>
        <label for="db_path">cards.cdb 路径</label>
        <input id="db_path" name="db_path" value="data/cards.cdb">
      </div>
      <div>
        <label for="top_k">Top K</label>
        <input id="top_k" name="top_k" type="number" min="1" max="50" value="10">
      </div>
      <div>
        <label for="rerank_candidates">Rerank candidates</label>
        <input id="rerank_candidates" name="rerank_candidates" type="number" min="1" max="200" value="20">
      </div>
    </div>
    <div class="checks">
      <label><input id="semantic" name="semantic" type="checkbox" checked> Chroma semantic</label>
      <label><input id="rerank" name="rerank" type="checkbox" checked> bge rerank</label>
      <label><input id="llm_rerank" name="llm_rerank" type="checkbox"> LLM judge rerank</label>
      <label><input id="llm" name="llm" type="checkbox"> DeepSeek LLM</label>
    </div>
    <button id="submit" type="submit">查询</button>
  </form>
  <section id="output" class="panel" hidden></section>
  <section class="panel">
    <h2>AI 裁定</h2>
    <form id="adjudication-form">
      <label for="scenario">具体情景</label>
      <textarea id="scenario" name="scenario">双方场上都只有一只怪兽的时候，对方发动破坏效果，我方怪兽可以代替破坏吗？</textarea>
      <div class="grid">
        <div>
          <label for="adjudication_db_path">cards.cdb 路径</label>
          <input id="adjudication_db_path" name="db_path" value="data/cards.cdb">
        </div>
        <div>
          <label for="ygocdb_cache_path">ygocdb cache</label>
          <input id="ygocdb_cache_path" name="ygocdb_cache_path" value="data/ygocdb">
        </div>
        <div>
          <label for="rules_epub_path">OCG rules EPUB</label>
          <input id="rules_epub_path" name="rules_epub_path" value="ocg-rule-readthedocs-io-zh-cn-latest.epub">
        </div>
      </div>
      <div class="checks">
        <label><input id="adjudication_llm" name="llm" type="checkbox"> DeepSeek LLM</label>
        <label><input id="adjudication_agent" name="agent" type="checkbox"> LangGraph agent</label>
        <label><input id="adjudication_trace" name="trace" type="checkbox"> trace</label>
        <label><input id="simulate" name="simulate" type="checkbox"> simulation</label>
      </div>
      <button id="adjudication-submit" type="submit">裁定</button>
    </form>
    <div id="adjudication-output" class="answer" hidden></div>
  </section>
</main>
<script>
const form = document.querySelector("#query-form");
const output = document.querySelector("#output");
const submit = document.querySelector("#submit");
const adjudicationForm = document.querySelector("#adjudication-form");
const adjudicationOutput = document.querySelector("#adjudication-output");
const adjudicationSubmit = document.querySelector("#adjudication-submit");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  submit.textContent = "查询中...";
  output.hidden = false;
  output.innerHTML = "<p class='muted'>正在检索，首次加载本地模型可能较慢。</p>";
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.semantic = document.querySelector("#semantic").checked;
  payload.rerank = document.querySelector("#rerank").checked;
  payload.llm_rerank = document.querySelector("#llm_rerank").checked;
  payload.llm = document.querySelector("#llm").checked;
  payload.top_k = Number(payload.top_k || 10);
  payload.rerank_candidates = Number(payload.rerank_candidates || 20);
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "请求失败");
    renderResult(data);
  } catch (error) {
    output.innerHTML = `<p class="warning">${escapeHtml(error.message)}</p>`;
  } finally {
    submit.disabled = false;
    submit.textContent = "查询";
  }
});

adjudicationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  adjudicationSubmit.disabled = true;
  adjudicationSubmit.textContent = "裁定中...";
  adjudicationOutput.hidden = false;
  adjudicationOutput.textContent = "正在收集裁定和规则证据...";
  const payload = Object.fromEntries(new FormData(adjudicationForm).entries());
  payload.llm = document.querySelector("#adjudication_llm").checked;
  payload.agent = document.querySelector("#adjudication_agent").checked;
  payload.trace = document.querySelector("#adjudication_trace").checked;
  payload.simulate = document.querySelector("#simulate").checked;
  try {
    const res = await fetch("/api/adjudicate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "请求失败");
    adjudicationOutput.innerHTML = `<h3>回答</h3><div>${escapeHtml(data.answer || "")}</div><h3>证据</h3>${renderEvidence(data.evidence || [])}${renderAgentTrace(data.agent_trace || [])}`;
  } catch (error) {
    adjudicationOutput.innerHTML = `<p class="warning">${escapeHtml(error.message)}</p>`;
  } finally {
    adjudicationSubmit.disabled = false;
    adjudicationSubmit.textContent = "裁定";
  }
});

function renderResult(data) {
  const warnings = (data.warnings || []).map(w => `<p class="warning">${escapeHtml(w)}</p>`).join("");
  const cards = (data.results || []).map((card, index) => `
    <div class="card">
      <div class="card-title">${index + 1}. ${escapeHtml(card.name)} <span class="muted">ID: ${card.card_id}, score: ${Number(card.score).toFixed(4)}</span></div>
      <p>${escapeHtml(card.source_text)}</p>
      <p class="muted">${escapeHtml(card.reason)}</p>
    </div>
  `).join("");
  output.innerHTML = `${warnings}<h2>回答</h2><div class="answer">${escapeHtml(data.answer || "")}</div><h2>候选卡</h2>${cards || "<p class='muted'>无候选结果</p>"}`;
}

const baseRenderResult = renderResult;
renderResult = function(data) {
  baseRenderResult(data);
  const filterInfo = renderFilterInfo(data);
  if (filterInfo) output.insertAdjacentHTML("afterbegin", filterInfo);
};

function renderFilterInfo(data) {
  const structuredQuery = data.structured_query || {};
  const diagnostics = data.filter_diagnostics || {};
  if (!structuredQuery.has_filters && !diagnostics.applied) return "";
  return `
    <div class="panel">
      <h2>结构化过滤</h2>
      <p class="muted">Effect query: <code>${escapeHtml(structuredQuery.effect_query || "")}</code></p>
      <p class="muted">Filters: <code>${escapeHtml(JSON.stringify(structuredQuery.filters || {}))}</code></p>
      <p class="muted">Candidates: ${Number(diagnostics.filtered_candidates || 0)} / ${Number(diagnostics.total_candidates || 0)}</p>
    </div>
  `;
}

function renderEvidence(items) {
  if (!items.length) return "<p class='muted'>无证据</p>";
  return items.map(item => `
    <div class="card">
      <div class="card-title">${escapeHtml(item.evidence_id)} <span class="muted">${escapeHtml(item.kind)} / ${escapeHtml(item.relationship)}</span></div>
      <p>${escapeHtml(item.title || "")}</p>
      <p class="muted">${escapeHtml(item.text || "")}</p>
    </div>
  `).join("");
}

function renderAgentTrace(items) {
  if (!items.length) return "";
  const rows = items.map((entry, index) => {
    const metadata = entry.metadata || {};
    const action = metadata.action || {};
    const observation = metadata.observation || {};
    const args = action.arguments ? JSON.stringify(action.arguments) : "";
    const evidenceIds = (observation.evidence_ids || metadata.evidence_ids || []).join(", ");
    return `
      <div class="card">
        <div class="card-title">${index + 1}. ${escapeHtml(entry.node || "")} <span class="muted">${escapeHtml(action.tool || "")}</span></div>
        <p>${escapeHtml(entry.summary || "")}</p>
        ${args ? `<p class="muted">args: <code>${escapeHtml(args)}</code></p>` : ""}
        ${observation.summary ? `<p class="muted">observation: ${escapeHtml(observation.summary)}</p>` : ""}
        ${evidenceIds ? `<p class="muted">evidence: ${escapeHtml(evidenceIds)}</p>` : ""}
      </div>
    `;
  }).join("");
  return `<h3>Agent Trace</h3>${rows}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[ch]));
}
</script>
</body>
</html>"""


def _default_query_handler(request: QueryRequest) -> QueryResponse:
    return execute_query(request, Settings.from_env())


def _default_translation_handler(request: TranslationRequest) -> TranslationResponse:
    return execute_translation(request, Settings.from_env())


def _default_adjudication_handler(request: AdjudicationRequest) -> AdjudicationResponse:
    return execute_adjudication(request, Settings.from_env())


def _default_assistant_handler(request: AssistantAgentRequest) -> AssistantAgentResponse:
    return execute_assistant_agent(request, Settings.from_env())


def _default_rift_handler(request: RiftQueryRequest) -> Mapping[str, Any]:
    return execute_rift_query(request, Settings.from_env())


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


async def _read_json_body(receive) -> dict[str, Any]:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object.")
    return payload


async def _send_html(send, content: str, status: int = 200) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/html; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": content.encode("utf-8")})


async def _send_response(send, status: int, payload: Mapping[str, Any]) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        }
    )


def _int_in_range(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _optional_int_in_range(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None or value == "":
        return None
    return _int_in_range(value, name, minimum=minimum, maximum=maximum)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_rerank_provider(payload: Mapping[str, Any]) -> str | None:
    provider = payload.get("rerank_provider")
    if provider is not None:
        return str(provider)
    if _as_bool(payload.get("llm_rerank", False)):
        return "llm"
    return None
