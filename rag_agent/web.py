from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from rag_agent.config import Settings
from rag_agent.query_service import QueryRequest, QueryResponse, execute_query

QueryHandler = Callable[[QueryRequest], QueryResponse]


def create_app(query_handler: QueryHandler | None = None):
    handler = query_handler or _default_query_handler

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
                await _send_response(send, 200, query_response_to_dict(response))
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
        use_llm=_as_bool(payload.get("llm", payload.get("use_llm", False))),
        rerank_candidates=_int_in_range(
            payload.get("rerank_candidates", 20),
            "rerank_candidates",
            minimum=1,
            maximum=200,
        ),
    )


def query_response_to_dict(response: QueryResponse) -> dict[str, Any]:
    return {
        "answer": response.answer,
        "results": response.results,
        "warnings": response.warnings,
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
      <label><input id="llm" name="llm" type="checkbox"> DeepSeek LLM</label>
    </div>
    <button id="submit" type="submit">查询</button>
  </form>
  <section id="output" class="panel" hidden></section>
</main>
<script>
const form = document.querySelector("#query-form");
const output = document.querySelector("#output");
const submit = document.querySelector("#submit");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  submit.textContent = "查询中...";
  output.hidden = false;
  output.innerHTML = "<p class='muted'>正在检索，首次加载本地模型可能较慢。</p>";
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.semantic = document.querySelector("#semantic").checked;
  payload.rerank = document.querySelector("#rerank").checked;
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
