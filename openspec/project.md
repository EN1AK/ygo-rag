# Project: Yu-Gi-Oh! Card RAG Agent

## Purpose

This project is a local Python application for querying Yu-Gi-Oh! card effects from a YGOPro `cards.cdb` database and returning cards with similar effect text.

The current implementation supports command-line usage and a local Web UI. It is designed for local development and local runtime assets rather than hosted multi-user deployment.

## Current Goals

- Load card records from a Chinese YGOPro `cards.cdb` SQLite database.
- Build a persistent local Chroma vector index over card effect documents.
- Retrieve candidate cards using sparse lexical search, optional dense semantic search, optional hybrid fusion, and optional local reranking.
- Optionally synthesize a Chinese answer with DeepSeek through a LangChain-compatible chat model.
- Preserve source card text in retrieval results so answers can be audited.
- Keep downloaded databases, Chroma indexes, local model caches, and credentials out of git.

## Technology Stack

- Language: Python 3.10+
- CLI: `argparse`
- Database reader: Python standard-library `sqlite3`
- Vector database: Chroma via `langchain-chroma`
- Embeddings: local `BAAI/bge-m3`
- Reranker: local `BAAI/bge-reranker-v2-m3`
- LLM: DeepSeek API through `langchain-openai` `ChatOpenAI`
- Sparse retrieval: custom Chinese character/bigram tokenizer and term-frequency scoring
- Web runtime: ASGI app served by `uvicorn`
- Tests: `pytest`

## Architecture

```text
User
  ├─ CLI: python -m rag_agent query
  └─ Web: POST /api/query
        │
        ▼
QueryRequest
        │
        ▼
query_service.execute_query()
        │
        ├─ load cards.cdb
        ├─ build sparse retriever
        ├─ optionally load Chroma + bge-m3 embeddings
        ├─ optionally load local reranker
        └─ optionally configure DeepSeek LLM
        │
        ▼
CardRagAgent.retrieve()
        │
        ├─ expand query with referenced card effect when an exact card name appears
        ├─ sparse search
        ├─ optional dense search
        ├─ reciprocal rank fusion
        └─ optional rerank
        │
        ▼
RetrievedCard[]
        │
        ├─ format retrieval-only answer
        └─ optional DeepSeek answer synthesis
```

## Runtime Configuration

Configuration is read from environment variables:

- `RAG_DATA_DIR`
- `RAG_CARDS_DB`
- `CHROMA_PERSIST_DIR`
- `RAG_DEVICE`
- `RAG_EMBEDDING_DEVICE`
- `RAG_RERANKER_DEVICE`
- `RAG_EMBEDDING_MODEL`
- `RAG_RERANKER_MODEL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `HF_HUB_OFFLINE`

Credentials are not read from files in the repository and must not be committed.

## Constraints

- Downloaded `cards.cdb` files are runtime data and must not be committed.
- Chroma persistence directories are runtime data and must not be committed.
- API keys, tokens, and credentials must not be committed.
- Tests use synthetic SQLite fixtures and must not require the live `cards.cdb` URL.
- Business logic should remain usable without DeepSeek when `--llm` is not enabled.
- Web UI is local-first and defaults to `127.0.0.1`.
- The current implementation is not a hosted multi-user service.

## Development Rules

- Keep parsing, retrieval, indexing, answer formatting, and UI boundaries separated.
- Prefer explicit dataclasses and typed function boundaries.
- Use `pathlib.Path` for filesystem paths.
- Preserve original card effect text in outputs where possible.
- Optional heavy dependencies should be loaded only when the related feature is used.
- Run tests before reporting implementation complete:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest
```

## Known Issues

- Similar-effect quality is currently based on lexical/dense/reranker behavior, not a structured model of card effect intent.
- Exact card-name matches are strongly boosted by sparse retrieval and may return the reference card itself.
- Chroma index validation currently checks document count, not source database hash, model version, or schema version.
- Web requests execute synchronous query work and may block during model loading or long retrieval.
- LLM output is free-form Chinese text, not a structured schema.

## Open Questions

- Should "similar effect" mean rules-text similarity, practical deck-building substitutability, or shared effect mechanism?
- Should the reference card be excluded from default "similar card" results?
- Should card type, legality, banlist, or format information affect ranking?
- Should the project require reproducible Chroma index manifests?
- Should the Web UI remain local-only or become a LAN/multi-user interface?
