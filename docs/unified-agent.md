# Unified Agent deployment (2026-10-12)

The production workspace is the server at `/home/ubuntu/qqbot`.
`ygo-rag.service` is the sole knowledge/chat backend on `127.0.0.1:7861`.
`kami-man.service` remains the QQ adapter; `rift-rag.service` is retired after
successful in-process verification. No training/simulator components are loaded.

## Request flow

QQ mention, translation alias, and Rift rules alias -> session-aware bot client
-> POST `/api/agent` -> conversation or professional tools. Ordinary chat is a
plain reply. Card/ruling results retain cards, references, ambiguity and warnings.

Tools: `search_cards` (YGO), `adjudicate` (YGO), `rift_search_cards`, `rift_rules`,
`translate_text`, `clarify`, `finish`. Ambiguous game/card mentions should be
clarified. Ruling results remain grounded in specialist evidence. A single
specialist ruling answer is preserved verbatim rather than rewritten.

## Conversation

Trusted bot constructs `qq:<bot>:group:<group>:user:<user>` session IDs. History is
stored in `data/agent_sessions.db` with SQLite WAL, 30 minute expiry, at most 10
turns and a 12000 character history budget. Same-session requests are serialized;
different sessions may proceed concurrently. Quotes include only the quoted bot
reply, never another user's whole history. Send `@神人 清空对话` to clear history.

## Runtime

- Systemd drop-in: `/etc/systemd/system/ygo-rag.service.d/unified-agent.conf`.
- `PYTHONPATH=/home/ubuntu/qqbot/riftsim/src` provides the `rift_rag` library.
- `RIFT_ENV_FILE=/home/ubuntu/qqbot/riftsim/.env` provides legacy specialist model
  settings without changing process-wide credentials. `RIFT_API_KEY`,
  `RIFT_BASE_URL`, `RIFT_LLM_MODEL`, `RIFT_DEEP_MODEL` are explicit overrides.
- `RIFTBOUND_CARDS_DB=/home/ubuntu/qqbot/kami-man/data/riftbound/cards_bilingual.db`
  shares the repaired card snapshot. Rules/vectors remain in riftsim's workspace.
- HTTP dispatch uses four worker threads and 220 second request budgets. Slow
  requests do not run on the ASGI event loop. At most two heavy specialist tasks
  run concurrently; overload returns a retry message. Timed-out work retains
  capacity until its worker exits; Python CPU work cannot be forcibly cancelled.
- Compatible CPU BGE models share a locked inference instance. YGO Chroma and
  Rift NumPy indices remain distinct. SQLite rule connections are per request.
- `/healthz` reports the unified service; it does not invoke a model.
- Legacy `/api/rift/query`, `/api/rift/api/query` and `rift-query` CLI now use the
  library too. `mode=oneshot` remains supported. `RIFTBOUND_RAG_TRANSPORT=http`
  explicitly selects the previous HTTP gateway for rollback.

## Verification and rollback

Run backend tests with `PYTHONPATH=/home/ubuntu/qqbot/riftsim/src .venv/bin/python
-m pytest tests -q`, Rift tests with `PYTHONPATH=src .venv/bin/python -m pytest
scripts/rag/tests -q`, and bot tests with `DRIVER=~none venv/bin/python -m pytest
tests -q`. Tests use fake LLMs/data; live HTTP probes additionally checked chat,
conversation recall, domain selection, source preservation and concurrent chat.

If rolling back, restore the previous tracked source in each repo, remove only
`unified-agent.conf`, and `systemctl daemon-reload`. Enable/start `rift-rag.service`
and restart `ygo-rag.service` and `kami-man.service`. Keep the repaired card DB,
image cache, original `.env` files and existing systemd drop-ins intact. The
riftsim script entry points remain compatibility wrappers for the package.
