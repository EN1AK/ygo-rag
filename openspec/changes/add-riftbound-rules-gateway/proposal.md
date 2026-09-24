# Add Riftbound Rules Gateway

## Why

The riftsim project provides a working Riftbound (符文战场) rules RAG as a standalone HTTP service on `127.0.0.1:7862`, and the kami-man bot currently calls it directly. The agreed target architecture is a single local RAG entry point: ygo-rag (`127.0.0.1:7860`) fronts all RAG capabilities and kami-man integrates only with ygo-rag. ygo-rag today has no Riftbound capability at all, so the bot cannot be switched over.

## What Changes

- Add a Riftbound rules QA gateway in ygo-rag: `POST /api/rift/query` accepts `{query, top_k?, mode?, trace?}` and forwards to the riftsim `rift-rag` service, returning its contract response (`answer`, `warnings`, `sources`) with additive v2 fields (`resolved_cards`, `coverage`, `exhausted`, `mode`, `trace`) passed through when present.
- The route path and request/response contract stay compatible with kami-man's existing rift client, so the bot switches endpoints configuration-only (`RIFT_RAG_BASE_URL=http://127.0.0.1:7860/api/rift`) with no kami-man code change.
- Add upstream failure mapping: unreachable/timeout/5xx/invalid-response from the rift-rag service become deterministic JSON errors (502/504) instead of generic 500s.
- Add runtime configuration for the upstream base URL and timeout.
- Add a `rift-query` CLI subcommand for local verification without the bot.

## Capabilities

### New Capabilities

- `riftbound-rules-gateway`: Proxying Riftbound rules QA requests to the riftsim `rift-rag` HTTP service, including request validation, additive v2 field pass-through, upstream timeout/error mapping, and contract compatibility with the existing kami-man rift client.

### Modified Capabilities

- `web-ui`: The web app gains a new JSON route `POST /api/rift/query` for Riftbound rules QA with its own validation and upstream-error semantics.
- `runtime-configuration`: New environment settings `RIFTBOUND_RAG_BASE_URL` and `RIFTBOUND_RAG_TIMEOUT_SECONDS` configure the upstream rift-rag service endpoint.
- `cli-interface`: New `rift-query` subcommand queries the Riftbound rules RAG through the gateway client from the command line.

## Impact

- Affected code (this repo):
  - New `rag_agent/rift_gateway.py` (upstream client, request/response models, validation, error mapping).
  - `rag_agent/web.py` (new `/api/rift/query` route and parsing).
  - `rag_agent/config.py` (two new settings).
  - `rag_agent/__main__.py` (new `rift-query` subcommand).
  - New `tests/test_rift_gateway.py` plus route cases in `tests/test_web.py`.
- Affected external systems:
  - riftsim `rift-rag` service must run and be reachable (unchanged; consumed read-only over HTTP).
  - kami-man bot: configuration-only switch to the ygo-rag entry; kami-man's fixed failure-message wording mentioning port 7862 is cosmetic and out of scope here.
- Out of scope:
  - riftsim-side agent mode / contract v2 server work (`agent_loop.py`, server v2). The gateway is forward-compatible: v2 fields pass through whenever the upstream starts emitting them.
  - Any change to ygo-rag's existing YGO query/translation/adjudication/agent routes.
