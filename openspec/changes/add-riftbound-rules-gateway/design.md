# Design: Riftbound Rules Gateway

## Context

See proposal.md - Why for motivation. Current state and constraints:

- **ygo-rag web layer** (`rag_agent/web.py`): bare ASGI app, no framework. Routes `POST /api/query|/api/translate|/api/adjudicate|/api/agent`; `ValueError` → 400, unexpected `Exception` → 500, unknown path → 404 JSON. Handlers are injected into `create_app()` for testability, and handlers run synchronously (accepted local-first limitation noted in project.md).
- **riftsim rift-rag service** (`127.0.0.1:7862`): `POST /api/query` accepts `{"query": str, "top_k": 1..20}` and returns `{"answer": str, "warnings": [str], "sources": [{"rule_id", "topic"}]}`; errors are JSON `{"error": str}` with 400/500. Extra request keys are ignored by the upstream validator. A v2 contract (`mode`, `trace` in; `resolved_cards`, `coverage`, `exhausted`, `mode`, `trace` out) is planned upstream but not yet implemented.
- **kami-man rift client** (`services/rift_rag_api.py`): posts to `f"{base_url}/api/query"` with `mode`/`trace` keys, tolerates missing v2 response fields, default timeout 240s. Its base URL is env-configurable (`RIFT_RAG_BASE_URL`).
- **Repo dependencies**: `requests` is already a dependency; `httpx` is not. Configuration flows through `Settings.from_env()` in `rag_agent/config.py`.

## Goals / Non-Goals

**Goals:**

- ygo-rag becomes the single local RAG entry: kami-man switches to it configuration-only (no kami-man code change).
- Deterministic upstream failure mapping (502/504 with JSON error) instead of generic 500s.
- Forward-compatible with the upstream v2 contract: v2 fields pass through untouched whenever the upstream starts emitting them.
- Testable without a live riftsim service (stubbed transport), mirroring the repo's synthetic-fixture test rules.

**Non-Goals:**

- Any change inside the riftsim repo (agent mode, upstream v2 server work).
- In-process Riftbound retrieval (reading `rules.db` or importing riftsim code).
- Changes to existing YGO routes, the Web UI page, or kami-man code (its fixed failure copy mentioning port 7862 is a later kami-man-side change).
- Async IO, streaming, caching, auth, or multi-user concerns (local-first).

## Decisions

### 1. HTTP proxy, not in-process embedding

The gateway calls the rift-rag service over HTTP. Riftsim remains the single owner of its rules DB, vector index, and ~3 GB embedding/LLM stack; ygo-rag gains zero heavy dependencies. Alternatives considered: importing riftsim modules or reading `rules.db` directly — rejected for cross-repo coupling and duplicated model/runtime management.

### 2. New module `rag_agent/rift_gateway.py` with injected transport

All gateway logic lives in one module: request model + validation, error taxonomy (`RiftGatewayError` with categories `validation` / `unavailable` / `timeout` / `upstream`), upstream call, and response assembly. The HTTP call takes an injectable request callable so tests substitute a stub instead of monkeypatching; this matches the repo's handler-injection pattern in `create_app()`.

### 3. Route path `/api/rift/query`

kami-man builds its URL as `f"{base_url}/api/query"`. Pointing its base URL at `http://127.0.0.1:7860/api/rift` therefore reaches `POST /api/rift/query` with **zero kami-man code change**. Alternative considered: a `game` discriminator on the existing `/api/query` — rejected because it overloads the YGO endpoint's contract and would require kami-man changes.

### 4. Synchronous `requests` client

`requests` is already a dependency and the web app's handlers already run synchronously (a documented local-first trade-off in project.md). Introducing `httpx`/async for one proxied call adds a dependency for no user-visible benefit at this scale.

### 5. Gateway-side validation mirroring the upstream contract

`query` (non-empty string), `top_k` (int, 1..20, default 6), `mode` (`agent`/`oneshot`), `trace` (bool) are validated/defaults applied before any upstream call, so callers get 400s without a network round trip and the upstream contract stays the single source for valid payloads. `mode`/`trace` are forwarded only when explicitly provided, keeping the wire body identical to today's legacy contract by default.

### 6. Contract-preserving response assembly with v2 pass-through

The gateway normalizes and verifies the required fields (`answer` str, `warnings` list, `sources` list of `{rule_id, topic}`), then copies the known optional v2 keys (`resolved_cards`, `coverage`, `exhausted`, `mode`, `trace`) only when present upstream — it never synthesizes them. Alternative considered: byte-level pass-through of the upstream body — rejected because malformed upstream success bodies would then surface to the bot as confusing client-side parse failures instead of a deterministic 502.

### 7. Timeout budget 200s default

`RIFTBOUND_RAG_TIMEOUT_SECONDS` defaults to 200s: above the upstream's own 180s generation timeout (`RAG_SERVER_GENERATE_TIMEOUT`), below kami-man's 240s client timeout. Invalid or non-positive env values fall back to the default.

### 8. Error mapping at the web boundary

`rift_gateway` raises categorized errors; the web route maps them before the app's generic `Exception → 500` path: validation → 400, unavailable → 502, timeout → 504, upstream → 502 (message includes the upstream `error` text or status). kami-man's client maps any 5xx to its fixed internal-error copy, so bot-visible failure behavior is unchanged from today's direct-to-7862 setup.

## Risks / Trade-offs

- **Added network hop / single point of failure** → If either ygo-rag or riftsim is down, Riftbound QA fails; equal to today's bot→riftsim direct dependency plus one local hop. Rollback is a bot env-var change back to `http://127.0.0.1:7862`.
- **Timeout chain drift** (bot 240 > gateway 200 > upstream 180) → Ordering pins the budget; if upstream's generation timeout is reconfigured, the gateway default should be revisited.
- **Blocking sync call in the ASGI loop** → Consistent with existing ygo-rag local-first handlers; documented known limitation, accepted.
- **Upstream v2 adoption later** → Pass-through design plus contract tests pinning tolerated fields contain the drift; no gateway change needed when riftsim ships its agent mode.
- **kami-man failure copy still says "7862"** → Cosmetic; tracked as a follow-up kami-man change, not this one.

## Migration Plan

1. Land the gateway in ygo-rag with unit/route tests (stubbed transport; no live dependency).
2. Start riftsim `rag_server.py` (7862) and ygo-rag `web` (7860); curl-smoke `POST /api/rift/query` for both a plain rules question and an invalid request.
3. In kami-man deployment, set `RIFT_RAG_BASE_URL=http://127.0.0.1:7860/api/rift` and restart the bot; verify a group QA reply.
4. Rollback: repoint `RIFT_RAG_BASE_URL` to `http://127.0.0.1:7862` (or unset) and restart the bot.

## Open Questions

- Should the gateway also proxy a liveness check (e.g. `GET /api/rift/healthz`) for ops verification? Deferrable; the bot never calls it today.
- After riftsim ships contract v2, should kami-man's `RIFT_RAG_MODE`/`RIFT_RAG_TRACE` rollout be coordinated with updating its 7862 wording? Yes, as a later kami-man-side change.
