# Tasks: Add Riftbound Rules Gateway

## 1. Gateway client module

- [x] 1.1 Create `rag_agent/rift_gateway.py` with the request model and validation (`query` non-empty string; `top_k` int 1..20 default 6; `mode` in `agent`/`oneshot` when provided; `trace` bool when provided) and a categorized error taxonomy (`validation` / `unavailable` / `timeout` / `upstream`, message-carrying). Verify with `.\.venv\Scripts\python.exe -m pytest tests/test_rift_gateway.py -q` covering: minimal request defaults, explicit options verbatim, empty/whitespace query, non-integer and out-of-range `top_k`, invalid `mode`, and confirmation that no transport call happens on validation failure.
- [x] 1.2 Add the upstream call to `rift_gateway.py`: injectable sync transport built on `requests` by default, POSTing `{upstream_base_url}/api/query` with `query`/`top_k` plus `mode`/`trace` only when provided; connection errors → `unavailable`, `requests` timeout → `timeout`, non-200 → `upstream` including the upstream JSON `error` text or status. Verify the same pytest module covers each failure category via a stubbed transport (no live service).
- [x] 1.3 Add response assembly to `rift_gateway.py`: require string `answer`, list `warnings`, list `sources` of `{rule_id, topic}` (malformed success body → `upstream` error describing the malformed response); copy optional v2 keys `resolved_cards`, `coverage`, `exhausted`, `mode`, `trace` only when present, never synthesize them. Verify via pytest cases for legacy-contract pass-through, v2 field pass-through with values equal to upstream, and each malformed-body case.

## 2. Runtime configuration

- [x] 2.1 Extend `Settings` in `rag_agent/config.py` with `riftbound_rag_base_url` (env `RIFTBOUND_RAG_BASE_URL`, default `http://127.0.0.1:7862`) and `riftbound_rag_timeout_seconds` (env `RIFTBOUND_RAG_TIMEOUT_SECONDS`, default `200.0`, falling back to the default on non-numeric or non-positive values). Verify with `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q` plus new cases for default, override, and invalid timeout fallback.

## 3. Web route

- [x] 3.1 Add `POST /api/rift/query` to `rag_agent/web.py` following the existing handler-injection pattern: parse + validate the JSON body through the gateway request model (400 JSON error on validation failure), execute the gateway handler, and map gateway error categories to `400` (validation), `502` (unavailable and upstream), `504` (timeout) JSON responses instead of the generic 500. Verify with `.\.venv\Scripts\python.exe -m pytest tests/test_web.py -q` plus new route cases: valid → 200 pass-through with a stub handler, invalid body → 400, each upstream failure category → 502/504, unknown `/api/rift/*` path still → 404.

## 4. CLI subcommand

- [x] 4.1 Add a `rift-query` subcommand to `rag_agent/__main__.py` (positional question, `--top-k`, `--mode {agent,oneshot}`, `--trace`) that runs the gateway client and prints the answer, warnings, and each source `rule_id`; gateway failures surface through the existing `error:` stderr + exit code `1` path. Verify with `.\.venv\Scripts\python.exe -m rag_agent rift-query --help` listing the options, and `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -q` with new cases for success output formatting and failure exit code (stubbed transport).

## 5. Verification

- [x] 5.1 Run the full suite `.\.venv\Scripts\python.exe -m pytest` and `openspec validate add-riftbound-rules-gateway --strict`; confirm all pass with no regressions.（结果：351 passed；validate --strict 通过）
- [x] 5.2 Live smoke（存档于 `scripts/rift_gateway_smoke_*.out.json`）：riftsim `rag_server.py --preload` 于 7862 + ygo-rag `web` 于 7860，三场景验证通过——200（`{"query": "反制堆叠有上限吗"}`，v2 字段 mode/exhausted/coverage/resolved_cards 原样透传）、400（空 query，`{"error": "query must be a non-empty string."}`）、502（停掉上游后返回 `upstream unavailable` + 连接拒绝详情）。kami-man 接入方式：将环境变量 `RIFT_RAG_API_URL` 设置为 `http://127.0.0.1:7860/api/rift/query`（客户端使用完整 URL 环境变量，零代码改动；本条修正了任务原文中关于 URL 拼接的错误表述）。
