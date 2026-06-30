# Spec: Yu-Gi-Oh! Card Effect Similarity RAG Agent

## Assumptions

1. The first deliverable is a local Python application, not a hosted web service.
2. The initial interface is CLI-first; an HTTP API can be added later after retrieval quality is proven.
3. The knowledge base source is the SQLite `cards.cdb` file at `https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb`.
4. The agent should answer in Chinese by default because the source database is `zh-CN`.
5. Similarity should be based primarily on card effect text, with card name, type, race, attribute, and set metadata used as secondary context.
6. The system may download and cache the database locally under the repository, but should not commit the downloaded `.cdb` file.
7. The agent uses DeepSeek's LLM API for answer synthesis and reasoning over retrieved context.
8. The retrieval/orchestration framework is LangChain.
9. Embeddings use local `bge-m3`.
10. Vector storage uses Chroma.
11. Retrieval uses hybrid search followed by local `bge-reranker-v2-m3` reranking.
12. Target example: input such as `有没有效果类似“我身作盾”的卡` should return cards with similar protective/negation/destruction-prevention effects, explain why they are similar, and include enough card metadata to verify the result.
13. A local Web UI is in scope as a convenience layer over the same retrieval pipeline; it must not introduce a separate query implementation or store API keys in the browser.

Correct these assumptions before implementation if any are wrong.

## Objective

Build an agent with RAG functionality for querying Yu-Gi-Oh! cards by semantically similar effects.

Primary user story:

- As a user, I can ask a natural-language Chinese question such as `有没有效果类似“我身作盾”的卡`, and the agent returns candidate cards with similar effects, concise explanations, and source text snippets from the card database.

Acceptance criteria:

- The agent can ingest `cards.cdb` into a local searchable knowledge base.
- The agent can resolve an exact or fuzzy card name query, e.g. `我身作盾`.
- The agent can perform semantic retrieval over effect text.
- The agent can produce ranked results with:
  - card name
  - card ID
  - card type/category if available
  - original effect text snippet
  - short explanation of similarity
  - similarity score or rank
- The answer should distinguish between:
  - cards with genuinely similar effect semantics
  - cards that only share superficial keywords
- The retrieval pipeline should be testable without requiring network access after the database and model/index are cached.

## Tech Stack

Proposed stack:

- Language: Python 3.10+
- Database reader: `sqlite3` from the Python standard library
- CLI: `argparse`
- Data validation/models: `dataclasses` initially; upgrade to Pydantic only if API boundaries become complex
- RAG framework: LangChain
- LLM provider: DeepSeek API via LangChain-compatible chat model integration
- Embeddings: local `bge-m3`
- Vector database: Chroma
- Retrieval: hybrid search combining dense vector similarity and sparse/lexical retrieval
- Reranker: local `bge-reranker-v2-m3`
- Embeddings:
  - local `bge-m3`, loaded from local cache/model path when available
  - no remote embedding provider by default
- Tests: `pytest`

Dependency decision:

- Adding `langchain`, `langchain-openai` or DeepSeek-compatible integration packages, `chromadb`, `langchain-chroma`, `sentence-transformers`, `FlagEmbedding`, `rank-bm25`, or other packages is in scope for this implementation.
- DeepSeek API credentials must be supplied through environment variables, not committed files.
- Local model downloads or model paths for `bge-m3` and `bge-reranker-v2-m3` must be explicit and cacheable.

## Commands

Initial commands to support:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m rag_agent download-db --url "https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb"
.\.venv\Scripts\python.exe -m rag_agent inspect-db
.\.venv\Scripts\python.exe -m rag_agent build-index
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡"
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡" --semantic --rerank --llm
.\.venv\Scripts\python.exe -m rag_agent web --host 127.0.0.1 --port 7860
```

Environment variables:

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:RAG_EMBEDDING_MODEL="BAAI/bge-m3"
$env:RAG_RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
$env:CHROMA_PERSIST_DIR="data/chroma"
```

Expected no-network commands after setup:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m rag_agent inspect-db
.\.venv\Scripts\python.exe -m rag_agent build-index
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡"
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡" --semantic --rerank
```

## Project Structure

```text
docs/
  spec-agent-rag.md           # This specification
rag_agent/
  __init__.py
  __main__.py                 # CLI entrypoint
  web.py                      # Local ASGI Web UI
  query_service.py            # Shared query orchestration for CLI and Web
  config.py                   # Paths and runtime configuration
  db.py                       # cards.cdb download/cache/schema/read logic
  cards.py                    # Card domain models and text normalization
  retrieval/
    __init__.py
    embeddings.py             # bge-m3 local embedding adapter
    sparse.py                 # Sparse/lexical retrieval signals
    vector.py                 # Chroma dense vector retrieval
    reranker.py               # bge-reranker-v2-m3 local reranker
    hybrid.py                 # Ranking fusion and reranking
  llm.py                      # DeepSeek/LangChain LLM configuration
  agent.py                    # LangChain query orchestration and answer composition
tests/
  test_db.py
  test_cards.py
  test_retrieval.py
  fixtures/
    mini_cards.cdb            # Tiny synthetic SQLite fixture, not full downloaded DB
data/
  .gitkeep
  cards.cdb                   # Downloaded runtime file; ignored by git
  index/                      # Generated runtime indexes; ignored by git
requirements.txt
.gitignore
```

## Code Style

Use small, typed functions with explicit inputs and outputs. Keep I/O at the edges and make parsing/retrieval logic unit-testable.

Example style:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    description: str


def normalize_effect_text(text: str) -> str:
    """Normalize card effect text for indexing without losing semantic content."""
    return " ".join(text.replace("\r\n", "\n").split())
```

Conventions:

- Prefer `pathlib.Path` over raw path strings.
- Use UTF-8 explicitly for text files.
- Do not mix database access, retrieval, and answer formatting in the same function.
- Keep generated data out of source control.
- Return structured data from retrieval; format user-facing text only at the CLI/API boundary.
- Keep the Web UI local-first by default and read provider credentials from environment variables only.

## Testing Strategy

Test levels:

- Unit tests:
  - text normalization
  - card row parsing
  - query intent parsing
  - lexical/vector scoring helpers
- Integration tests:
  - create a tiny synthetic `cards.cdb` fixture with `datas` and `texts` tables
  - ingest fixture
  - build a small index
  - query for a known card/effect and assert expected similar cards rank above unrelated cards
- CLI smoke tests:
  - `inspect-db` reports schema and row counts
  - `query` returns valid ranked output
- Web smoke tests:
  - `GET /` renders the query page
  - `POST /api/query` returns structured JSON with answer, candidates, and warnings

Coverage expectations:

- Core parsing/retrieval/agent modules should have meaningful tests before implementation is considered complete.
- Tests must not depend on the live remote `cards.cdb` URL.

## Boundaries

- Always:
  - Keep downloaded `cards.cdb` and generated indexes out of git.
  - Read DeepSeek API key from environment variables only.
  - Keep local model caches and Chroma storage out of git.
  - Use a synthetic mini database fixture for tests.
  - Preserve source card text in answers so retrieval can be audited.
  - Validate that expected YGOPro tables/columns exist before indexing.
  - Run tests before reporting implementation complete.

- Ask first:
  - Adding network-dependent embedding providers beyond local `bge-m3`.
  - Replacing Chroma with another vector store.
  - Changing from CLI-first to web/API-first.
  - Downloading large models or runtime assets outside the repository.
  - Persisting generated artifacts outside `D:\workspace\rag`.

- Never:
  - Commit the downloaded full `cards.cdb`.
  - Commit API keys, tokens, or credentials.
  - Treat LLM-generated explanations as ground truth without showing the source card text.
  - Delete user files or generated caches outside this project.

## Success Criteria

The feature is complete when:

1. `python -m rag_agent download-db --url ".../cards.cdb"` downloads or refreshes the local database cache.
2. `python -m rag_agent inspect-db` reports the database schema and confirms card text rows are readable.
3. `python -m rag_agent build-index` creates a reusable local index.
4. Chroma contains dense vectors and payload metadata for indexed cards.
5. Hybrid retrieval combines dense `bge-m3` vector results with sparse/lexical signals, then reranks candidates with local `bge-reranker-v2-m3`.
6. `python -m rag_agent query "有没有效果类似“我身作盾”的卡"` returns ranked candidate cards with explanations and source snippets.
7. Tests pass with a synthetic fixture and do not require live network access, except optional explicitly marked integration tests for DeepSeek/Qdrant/model availability.
8. The implementation handles missing database/index/model/Chroma files with clear error messages.

## Open Questions

1. Should the first user interface be CLI only, or do you want an HTTP API/UI from the start?
2. Should Chroma use local persistent mode under `data/chroma`, or connect to a separate Chroma server?
3. Is the expected output purely retrieval results, or should the agent also synthesize strategy-oriented explanations?
4. What recall/precision target is acceptable for the first version? For example: top 10 should include at least 3 obviously related cards for known benchmark queries.
5. Should card legality/banlist/format information be part of ranking, or is effect similarity the only criterion?

## Phase 2 Implementation Plan

### Components

1. Database ingestion
   - Download `cards.cdb` into `data/cards.cdb`.
   - Inspect SQLite schema and verify expected YGOPro tables.
   - Extract card records from `texts` and `datas`.
   - Normalize effect text while preserving source text for answer citations.

2. Document preparation
   - Convert cards into LangChain-compatible documents.
   - Store payload fields needed for filtering and answer rendering:
     - `card_id`
     - `name`
     - `desc`
     - `type`
     - `race`
     - `attribute`
     - `atk`
     - `def`
     - `level`

3. Local embedding
   - Load `bge-m3` locally.
   - Produce dense vectors for card documents.
   - Use deterministic batching and clear errors when the model is missing.

4. Chroma indexing
   - Create/recreate a collection for card effects.
   - Upsert dense vectors and payloads.
   - Store collection metadata/version so incompatible indexes can be rebuilt.

5. Sparse/lexical retrieval
   - Build a sparse retrieval signal from card names and effect text.
   - Use BM25 or Chroma-compatible metadata/payload retrieval depending on dependency fit.
   - Boost exact/fuzzy card-name matches when the query mentions a known card.

6. Hybrid retrieval
   - Retrieve candidate cards from:
     - dense Chroma vector search
     - sparse/lexical search
     - optional name-resolution seed card expansion
   - Merge candidates with reciprocal rank fusion or weighted score fusion.
   - Deduplicate by `card_id`.

7. Local reranking
   - Load `bge-reranker-v2-m3`.
   - Rerank top candidate card effects against the user query and/or resolved source card effect.
   - Return final top-k results.

8. LangChain agent/chain
   - Build a deterministic RAG chain:
     - parse user query
     - retrieve and rerank cards
     - call DeepSeek through LangChain
     - answer with ranked cards and source snippets
   - Keep retrieval result structure independent from LLM prose.

9. CLI
   - Commands:
     - `download-db`
     - `inspect-db`
     - `build-index`
     - `query`
   - Each command should fail with actionable messages for missing API key, model, DB, index, or Qdrant.

10. Tests
   - Synthetic SQLite fixture tests for ingestion.
   - Unit tests for normalization and document conversion.
   - Unit tests for hybrid merge/dedup/ranking logic.
   - CLI smoke tests where feasible.
   - Mark external integration tests separately if they require Qdrant, local models, or DeepSeek.

### Implementation Order

1. Create project skeleton, `.gitignore`, and dependency manifest.
2. Add SQLite fixture and ingestion tests.
3. Implement card parsing and document conversion.
4. Implement Chroma configuration and index command.
5. Implement local `bge-m3` embedding adapter.
6. Implement sparse retrieval and hybrid fusion.
7. Implement local reranker adapter.
8. Implement LangChain + DeepSeek answer chain.
9. Implement CLI commands end to end.
10. Add a local Web UI that reuses the same query service as the CLI.
11. Run tests and perform a real query against `我身作盾` if required runtime assets are available.

### Risks and Mitigations

- Chroma persistence/runtime compatibility
  - Mitigation: make `CHROMA_PERSIST_DIR` explicit, use local persistent mode by default, and fail early if the collection cannot be opened.

- Local model size/download friction
  - Mitigation: support environment-configured local model names/paths and document setup clearly.

- `cards.cdb` schema drift
  - Mitigation: inspect schema before reading and isolate schema-specific code in `db.py`.

- Hybrid search quality may be weak without tuning
  - Mitigation: keep fusion weights configurable and add benchmark queries.

- LLM may overstate similarity
  - Mitigation: force answer prompt to cite retrieved card text and avoid unsupported claims.

### Verification Checkpoints

1. `inspect-db` works against fixture and real DB.
2. `build-index` writes Chroma collection and reports indexed card count.
3. Retrieval returns structured candidates before any LLM call.
4. Reranker changes candidate order deterministically for a fixed query.
5. DeepSeek answer uses only provided retrieved context.
6. Full CLI query works for `有没有效果类似“我身作盾”的卡`.

## Phase 3 Tasks

- [x] Task: Add Python project skeleton and dependency manifest
  - Acceptance: package imports as `rag_agent`; CLI module exists; runtime/generated data is ignored.
  - Verify: `python -m rag_agent --help`
  - Files: `rag_agent/*`, `requirements.txt`, `.gitignore`

- [x] Task: Implement cards.cdb ingestion against a synthetic SQLite fixture
  - Acceptance: parser reads YGOPro-style `datas` and `texts` tables and returns typed card records.
  - Verify: `python -m pytest tests/test_db.py`
  - Files: `rag_agent/db.py`, `rag_agent/cards.py`, `tests/test_db.py`

- [x] Task: Implement card text normalization and document conversion
  - Acceptance: cards become retrieval documents with stable text and metadata.
  - Verify: `python -m pytest tests/test_cards.py`
  - Files: `rag_agent/cards.py`, `tests/test_cards.py`

- [x] Task: Implement deterministic hybrid ranking primitives
  - Acceptance: dense and sparse candidate lists merge by reciprocal rank fusion, deduplicate by card ID, and preserve metadata.
  - Verify: `python -m pytest tests/test_retrieval.py`
  - Files: `rag_agent/retrieval/hybrid.py`, `tests/test_retrieval.py`

- [x] Task: Add local model and Chroma adapters with import-time isolation
  - Acceptance: missing optional dependencies produce clear runtime errors only when the adapter is used.
  - Verify: `python -m pytest tests/test_config.py`
  - Files: `rag_agent/config.py`, `rag_agent/retrieval/embeddings.py`, `rag_agent/retrieval/reranker.py`, `rag_agent/retrieval/vector.py`

- [x] Task: Add LangChain + DeepSeek answer composition
  - Acceptance: DeepSeek API key is read from environment only; answer prompt is constrained to retrieved context.
  - Verify: unit tests with a fake LLM, external integration test optional.
  - Files: `rag_agent/llm.py`, `rag_agent/agent.py`, `tests/test_agent.py`

- [x] Task: Implement CLI commands
  - Acceptance: `download-db`, `inspect-db`, `build-index`, and `query` commands exist and provide actionable errors.
  - Verify: CLI smoke tests and manual `--help`.
  - Files: `rag_agent/__main__.py`, `tests/test_cli.py`

- [x] Task: Add local Web UI
  - Acceptance: `web` command starts a local browser UI; `POST /api/query` returns answer text, structured candidates, and warnings using the shared query pipeline.
  - Verify: Web smoke tests and CLI help tests.
  - Files: `rag_agent/web.py`, `rag_agent/query_service.py`, `rag_agent/__main__.py`, `tests/test_web.py`
