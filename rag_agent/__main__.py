from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from rag_agent.config import Settings
from rag_agent.db import inspect_cards_db, load_cards
from rag_agent.agent import CardRagAgent
from rag_agent.cards import card_to_document
from rag_agent.retrieval.embeddings import BgeM3Embeddings
from rag_agent.retrieval.reranker import BgeReranker, SubprocessReranker
from rag_agent.retrieval.vector import ChromaConfig, ChromaVectorStore
from rag_agent.llm import create_deepseek_chat_model


def download_db(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    print(f"Downloaded cards database to {destination}")
    return 0


def inspect_db(path: Path) -> int:
    info = inspect_cards_db(path)
    print(f"Database: {info.path}")
    for table, count in info.table_counts.items():
        print(f"{table}: {count}")
    return 0


def build_index(
    path: Path,
    settings: Settings,
    *,
    batch_size: int = 128,
    limit: int | None = None,
    reset: bool = False,
) -> int:
    cards = load_cards(path)
    if limit is not None:
        cards = cards[:limit]
    documents = [card_to_document(card) for card in cards]
    print(f"Loading embedding model: {settings.embedding_model}", flush=True)
    embeddings = BgeM3Embeddings(settings.embedding_model, device=settings.embedding_device)
    vector_store = ChromaVectorStore(
        ChromaConfig(settings.chroma_persist_dir), embeddings
    )
    def progress(done: int, total: int) -> None:
        print(f"Indexed {done}/{total} cards", flush=True)

    vector_store.build(
        documents,
        batch_size=batch_size,
        reset=reset,
        progress_callback=progress,
    )
    print(f"Indexed {len(documents)} cards into Chroma: {settings.chroma_persist_dir}")
    return 0


def query_cards(
    query: str,
    db_path: Path,
    top_k: int,
    settings: Settings,
    *,
    semantic: bool = False,
    rerank: bool = False,
    use_llm: bool = False,
    rerank_candidates: int = 20,
) -> int:
    cards = load_cards(db_path)
    embeddings = BgeM3Embeddings(settings.embedding_model, device=settings.embedding_device)

    dense_retriever = None
    if semantic:
        candidate_dense_retriever = ChromaVectorStore(
            ChromaConfig(settings.chroma_persist_dir), embeddings
        )
        indexed_count = candidate_dense_retriever.count()
        if indexed_count < len(cards):
            print(
                f"warning: Chroma index has {indexed_count} documents but cards DB has {len(cards)}; "
                "skipping dense retrieval until the full index is built.",
                file=sys.stderr,
            )
        else:
            dense_retriever = candidate_dense_retriever

    reranker = None
    if rerank:
        if settings.embedding_device.startswith("cuda"):
            reranker_device = (
                settings.reranker_device
                if settings.reranker_device_explicit
                else "auto"
            )
            reranker = SubprocessReranker(
                settings.reranker_model,
                device=reranker_device,
            )
        else:
            reranker = BgeReranker(settings.reranker_model, device=settings.reranker_device)
    llm = create_deepseek_chat_model(settings) if use_llm else None

    agent = CardRagAgent(
        cards,
        dense_retriever=dense_retriever,
        reranker=reranker,
        llm=llm,
    )
    print(agent.query(query, top_k=top_k, rerank_candidates=rerank_candidates))
    return 0


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="rag_agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-db", help="download cards.cdb")
    download.add_argument("--url", required=True)
    download.add_argument("--out", type=Path, default=settings.cards_db_path)

    inspect = subparsers.add_parser("inspect-db", help="inspect cards.cdb schema")
    inspect.add_argument("--db", type=Path, default=settings.cards_db_path)

    build = subparsers.add_parser("build-index", help="build local Chroma index")
    build.add_argument("--db", type=Path, default=settings.cards_db_path)
    build.add_argument("--batch-size", type=int, default=128)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--reset", action="store_true")

    query = subparsers.add_parser("query", help="query similar card effects")
    query.add_argument("query")
    query.add_argument("--db", type=Path, default=settings.cards_db_path)
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--rerank-candidates", type=int, default=20)
    query.add_argument(
        "--semantic",
        action="store_true",
        help="include Chroma dense vector retrieval using local bge-m3",
    )
    query.add_argument(
        "--rerank",
        action="store_true",
        help="rerank hybrid candidates using local bge-reranker-v2-m3",
    )
    query.add_argument(
        "--llm",
        action="store_true",
        help="synthesize final answer with DeepSeek through LangChain",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download-db":
            return download_db(args.url, args.out)
        if args.command == "inspect-db":
            return inspect_db(args.db)
        if args.command == "build-index":
            return build_index(
                args.db,
                settings,
                batch_size=args.batch_size,
                limit=args.limit,
                reset=args.reset,
            )
        if args.command == "query":
            return query_cards(
                args.query,
                args.db,
                args.top_k,
                settings,
                semantic=args.semantic,
                rerank=args.rerank,
                use_llm=args.llm,
                rerank_candidates=args.rerank_candidates,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
