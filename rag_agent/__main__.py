from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from rag_agent.config import Settings
from rag_agent.db import inspect_cards_db, load_cards
from rag_agent.cards import card_to_document
from rag_agent.retrieval.embeddings import BgeM3Embeddings
from rag_agent.retrieval.reranker import BgeReranker, LlmReranker, SubprocessReranker
from rag_agent.retrieval.vector import ChromaConfig, ChromaVectorStore
from rag_agent.query_service import QueryRequest, execute_query
from rag_agent.translation_service import TranslationRequest, execute_translation
from rag_agent.assistant_agent import (
    AssistantAgentRequest,
    execute_assistant_agent,
)
from rag_agent.adjudication_service import (
    AdjudicationRequest,
    adjudication_response_to_dict,
    execute_adjudication,
    render_adjudication_answer,
)
from rag_agent.adjudication import YgocdbCache
from rag_agent.ruling_index import (
    PersistentRulingRetriever,
    RulingVectorStore,
    build_persistent_ruling_index,
    inspect_persistent_ruling_index,
    load_rule_documents_from_epub,
    load_ygocdb_ruling_documents_from_cache,
    normalize_ruling_rerank_provider,
    validate_manifest,
)
from rag_agent.llm import create_deepseek_chat_model


os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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


def download_rulings(
    db_path: Path,
    cache_path: Path,
    *,
    limit: int | None = None,
    delay_seconds: float = 0.1,
    timeout_seconds: float = 30.0,
    force: bool = False,
    fail_fast: bool = False,
    progress_every: int = 100,
    use_proxy: bool = False,
) -> int:
    cards = load_cards(db_path)
    if limit is not None:
        cards = cards[:limit]
    cache = YgocdbCache(cache_path)
    if cache_path.suffix:
        raise ValueError("download-rulings requires a cache directory, not a JSON file.")

    total = len(cards)
    downloaded = 0
    skipped = 0
    failed = 0
    cache_path.mkdir(parents=True, exist_ok=True)
    for index, card in enumerate(cards, start=1):
        if not force and cache.load_card_payload(card.card_id) is not None:
            skipped += 1
            if progress_every <= 1:
                print(
                    f"[{index}/{total}] skip {card.card_id} {card.name}",
                    flush=True,
                )
            elif index % progress_every == 0:
                print(
                    f"[{index}/{total}] progress downloaded={downloaded} "
                    f"skipped={skipped} failed={failed}",
                    flush=True,
                )
            continue
        try:
            payload = cache.fetch_card_payload(
                card.card_id,
                timeout=timeout_seconds,
                use_proxy=use_proxy,
            )
            cache.save_card_payload(card.card_id, payload)
            downloaded += 1
            faq_count = len(payload.get("faqs") or [])
            has_supplement = bool(payload.get("supplement"))
            if progress_every <= 1 or index % progress_every == 0:
                print(
                    f"[{index}/{total}] saved {card.card_id} {card.name} "
                    f"faqs={faq_count} supplement={has_supplement} "
                    f"downloaded={downloaded} skipped={skipped} failed={failed}",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(
                f"warning: failed {card.card_id} {card.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if fail_fast:
                raise
        if delay_seconds > 0 and index < total:
            time.sleep(delay_seconds)
        elif progress_every > 1 and index % progress_every == 0:
            print(
                f"[{index}/{total}] progress downloaded={downloaded} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )
    print(
        f"Ruling download complete: downloaded={downloaded}, "
        f"skipped={skipped}, failed={failed}, cache={cache_path}"
    )
    return 1 if failed and fail_fast else 0


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


def build_ruling_index_command(
    db_path: Path,
    settings: Settings,
    *,
    ygocdb_cache_path: Path | None = None,
    rules_epub_path: Path | None = None,
    faq_only: bool = False,
    rules_only: bool = False,
    batch_size: int = 128,
    limit: int | None = None,
    reset: bool = False,
) -> int:
    cards = load_cards(db_path)
    embeddings = BgeM3Embeddings(
        settings.embedding_model,
        device=settings.embedding_device,
    )
    build_faq = not rules_only
    build_rules = not faq_only
    if not build_faq and not build_rules:
        raise ValueError("At least one ruling index kind must be selected.")

    if build_faq:
        cache_path = ygocdb_cache_path or settings.ygocdb_cache_path
        documents, source_count = load_ygocdb_ruling_documents_from_cache(
            cache_path,
            cards,
            limit=limit,
        )
        print(
            f"Building FAQ ruling index from {source_count} cached payloads "
            f"({len(documents)} documents)...",
            flush=True,
        )

        def faq_progress(done: int, total: int) -> None:
            print(f"Indexed FAQ ruling documents {done}/{total}", flush=True)

        result = build_persistent_ruling_index(
            documents,
            index_kind="faq",
            persist_dir=settings.ruling_faq_index_path,
            collection_name=settings.ruling_faq_collection_name,
            manifest_path=settings.ruling_faq_manifest_path,
            source_paths=[cache_path],
            source_count=source_count,
            embedding_model=settings.embedding_model,
            embedding_function=embeddings,
            batch_size=batch_size,
            reset=reset,
            progress_callback=faq_progress,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if build_rules:
        epub_path = rules_epub_path or settings.ocg_rules_epub_path
        documents, source_count = load_rule_documents_from_epub(
            epub_path,
            limit=limit,
        )
        print(
            f"Building OCG rules index from {epub_path} "
            f"({len(documents)} documents)...",
            flush=True,
        )

        def rules_progress(done: int, total: int) -> None:
            print(f"Indexed OCG rules documents {done}/{total}", flush=True)

        result = build_persistent_ruling_index(
            documents,
            index_kind="rules",
            persist_dir=settings.ruling_rules_index_path,
            collection_name=settings.ruling_rules_collection_name,
            manifest_path=settings.ruling_rules_manifest_path,
            source_paths=[epub_path],
            source_count=source_count,
            embedding_model=settings.embedding_model,
            embedding_function=embeddings,
            batch_size=batch_size,
            reset=reset,
            progress_callback=rules_progress,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def inspect_ruling_index_command(settings: Settings, *, json_output: bool = False) -> int:
    embeddings = BgeM3Embeddings(
        settings.embedding_model,
        device=settings.embedding_device,
        show_progress=False,
    )
    inspections = [
        inspect_persistent_ruling_index(
            index_kind="faq",
            persist_dir=settings.ruling_faq_index_path,
            collection_name=settings.ruling_faq_collection_name,
            manifest_path=settings.ruling_faq_manifest_path,
            embedding_model=settings.embedding_model,
            embedding_function=embeddings,
        ),
        inspect_persistent_ruling_index(
            index_kind="rules",
            persist_dir=settings.ruling_rules_index_path,
            collection_name=settings.ruling_rules_collection_name,
            manifest_path=settings.ruling_rules_manifest_path,
            embedding_model=settings.embedding_model,
            embedding_function=embeddings,
        ),
    ]
    payload = [inspection.to_dict() for inspection in inspections]
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for item in payload:
        print(f"{item['index_kind']}: usable={item['usable']} count={item['collection_count']}")
        for warning in item["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


def query_rulings_command(
    query: str,
    db_path: Path,
    settings: Settings,
    *,
    ygocdb_cache_path: Path | None = None,
    rules_epub_path: Path | None = None,
    semantic: bool = False,
    rerank: bool = False,
    llm_rerank: bool = False,
    top_k: int = 10,
    fused_top_k: int | None = None,
    json_output: bool = False,
) -> int:
    cards = load_cards(db_path)
    faq_documents, _faq_sources = load_ygocdb_ruling_documents_from_cache(
        ygocdb_cache_path or settings.ygocdb_cache_path,
        cards,
    )
    rule_documents, _rule_sources = load_rule_documents_from_epub(
        rules_epub_path or settings.ocg_rules_epub_path
    )
    documents = [*faq_documents, *rule_documents]
    warnings: list[str] = []
    embeddings = None
    dense_store = None
    manifest_status = None
    if semantic:
        embeddings = BgeM3Embeddings(
            settings.embedding_model,
            device=settings.embedding_device,
            show_progress=False,
        )
        manifest_status = validate_manifest(
            settings.ruling_faq_manifest_path,
            index_kind="faq",
            collection_name=settings.ruling_faq_collection_name,
            embedding_model=settings.embedding_model,
        )
        if manifest_status.usable:
            dense_store = RulingVectorStore(
                settings.ruling_faq_index_path,
                settings.ruling_faq_collection_name,
                embeddings,
            )
        else:
            warnings.extend(manifest_status.warnings)
    rerank_provider = "llm" if llm_rerank else "local" if rerank else settings.ruling_rerank_provider
    local_reranker = None
    llm = None
    provider = normalize_ruling_rerank_provider(rerank_provider)
    if provider == "local":
        if settings.embedding_device.startswith("cuda"):
            reranker_device = (
                settings.reranker_device
                if settings.reranker_device_explicit
                else "auto"
            )
            local_reranker = SubprocessReranker(settings.reranker_model, device=reranker_device)
        else:
            local_reranker = BgeReranker(settings.reranker_model, device=settings.reranker_device)
    elif provider == "llm":
        llm = create_deepseek_chat_model(settings)
    retriever = PersistentRulingRetriever(
        documents,
        dense_store=dense_store,
        manifest_status=manifest_status,
        local_reranker=local_reranker,
        llm=llm,
        llm_max_candidates=settings.llm_rerank_max_candidates,
        warning_callback=warnings.append,
    )
    result = retriever.search(
        query,
        sparse_top_k=settings.ruling_sparse_top_k,
        dense_top_k=settings.ruling_dense_top_k,
        fused_top_k=fused_top_k or settings.ruling_fused_top_k,
        final_top_k=top_k,
        semantic=semantic,
        rerank_provider=provider,
    )
    warnings.extend(result.warnings)
    payload = result.to_dict() | {"warnings": warnings}
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for index, candidate in enumerate(result.candidates, start=1):
        print(
            f"{index}. {candidate.evidence_id} "
            f"{candidate.document.kind}/{candidate.relationship} "
            f"score={candidate.score:.4f}"
        )
        print(candidate.document.title)
        print(candidate.document.text[:240].replace("\n", " "))
    return 0


def query_cards(
    query: str,
    db_path: Path,
    top_k: int,
    settings: Settings,
    *,
    semantic: bool = False,
    rerank: bool = False,
    llm_rerank: bool = False,
    use_llm: bool = False,
    rerank_candidates: int = 20,
) -> int:
    if llm_rerank:
        print(
            "Calling DeepSeek LLM judge rerank; this may take a while...",
            file=sys.stderr,
            flush=True,
        )
    response = execute_query(
        QueryRequest(
            query=query,
            db_path=db_path,
            top_k=top_k,
            semantic=semantic,
            rerank=rerank,
            rerank_provider="llm" if llm_rerank else None,
            use_llm=use_llm,
            rerank_candidates=rerank_candidates,
        ),
        settings,
    )
    for warning in response.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if response.structured_query and response.structured_query.get("has_filters"):
        print("Structured filters:")
        print(json.dumps(response.structured_query, ensure_ascii=False, indent=2))
        if response.filter_diagnostics:
            print(json.dumps(response.filter_diagnostics, ensure_ascii=False, indent=2))
    print(response.answer)
    return 0


def translate_text(
    text: str,
    settings: Settings,
    *,
    source_lang: str = "auto",
    target_lang: str = "zh-CN",
    structured_max_block_chars: int | None = None,
) -> int:
    response = execute_translation(
        TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            structured_max_block_chars=structured_max_block_chars,
        ),
        settings,
    )
    for warning in response.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(response.translation)
    return 0


def adjudicate_scenario(
    scenario: str,
    db_path: Path,
    settings: Settings,
    *,
    ygocdb_cache_path: Path | None = None,
    rules_epub_path: Path | None = None,
    use_llm: bool = False,
    simulate: bool = False,
    allow_ygocdb_fetch: bool = False,
    use_agent: bool = False,
    include_trace: bool = False,
    show_progress: bool = False,
    output_format: str = "both",
) -> int:
    progress_callback = None
    if show_progress:
        def report_progress(event) -> None:
            print(event.message, file=sys.stderr, flush=True)

        progress_callback = report_progress

    response = execute_adjudication(
        AdjudicationRequest(
            scenario=scenario,
            db_path=db_path,
            ygocdb_cache_path=ygocdb_cache_path,
            rules_epub_path=rules_epub_path,
            use_llm=use_llm,
            simulate=simulate,
            allow_ygocdb_fetch=allow_ygocdb_fetch,
            use_agent=use_agent,
            include_trace=include_trace,
            progress_callback=progress_callback,
        ),
        settings,
    )
    for warning in response.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if output_format in {"both", "natural"}:
        print(render_adjudication_answer(response.answer))
    if output_format in {"both", "json"}:
        print(json.dumps(adjudication_response_to_dict(response), ensure_ascii=False, indent=2))
    return 0


def run_assistant_agent(
    text: str,
    db_path: Path,
    settings: Settings,
    *,
    semantic: bool = False,
    rerank: bool = False,
    llm_rerank: bool = False,
    max_steps: int = 6,
    ygocdb_cache_path: Path | None = None,
    rules_epub_path: Path | None = None,
    include_trace: bool = False,
    show_progress: bool = False,
    output_format: str = "both",
) -> int:
    progress_callback = None
    if show_progress:
        def report_progress(event: dict[str, object]) -> None:
            print(str(event.get("message", "")), file=sys.stderr, flush=True)

        progress_callback = report_progress

    response = execute_assistant_agent(
        AssistantAgentRequest(
            text=text,
            db_path=db_path,
            semantic=semantic,
            rerank=rerank,
            rerank_provider="llm" if llm_rerank else None,
            max_steps=max_steps,
            ygocdb_cache_path=ygocdb_cache_path,
            rules_epub_path=rules_epub_path,
            include_specialist_trace=include_trace,
        ),
        settings,
        progress_callback=progress_callback,
    )
    for warning in response.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if output_format in {"both", "natural"}:
        print(response.answer)
    if output_format in {"both", "json"}:
        print(
            json.dumps(
                response.to_dict(include_trace=include_trace),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


def run_web(host: str, port: int) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required for the web UI. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    from rag_agent.web import create_app

    uvicorn.run(create_app(), host=host, port=port)
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

    ruling_build = subparsers.add_parser(
        "build-ruling-index",
        help="build persistent ygocdb FAQ and OCG rules indexes",
    )
    ruling_build.add_argument("--db", type=Path, default=settings.cards_db_path)
    ruling_build.add_argument(
        "--ygocdb-cache",
        type=Path,
        default=settings.ygocdb_cache_path,
        help="local ygocdb API cache directory",
    )
    ruling_build.add_argument(
        "--rules-epub",
        type=Path,
        default=settings.ocg_rules_epub_path,
        help="local OCG rules EPUB path",
    )
    ruling_build.add_argument("--faq-only", action="store_true")
    ruling_build.add_argument("--rules-only", action="store_true")
    ruling_build.add_argument("--batch-size", type=int, default=128)
    ruling_build.add_argument("--limit", type=int, default=None)
    ruling_build.add_argument("--reset", action="store_true")

    ruling_inspect = subparsers.add_parser(
        "inspect-ruling-index",
        help="inspect persistent ruling index manifests and counts",
    )
    ruling_inspect.add_argument("--json", action="store_true")

    ruling_query = subparsers.add_parser(
        "query-rulings",
        help="query persistent ruling FAQ/rules evidence",
    )
    ruling_query.add_argument("query")
    ruling_query.add_argument("--db", type=Path, default=settings.cards_db_path)
    ruling_query.add_argument("--ygocdb-cache", type=Path, default=settings.ygocdb_cache_path)
    ruling_query.add_argument("--rules-epub", type=Path, default=settings.ocg_rules_epub_path)
    ruling_query.add_argument("--semantic", action="store_true")
    ruling_query.add_argument("--rerank", action="store_true")
    ruling_query.add_argument("--llm-rerank", action="store_true")
    ruling_query.add_argument("--top-k", type=int, default=settings.ruling_final_top_k)
    ruling_query.add_argument("--fused-top-k", type=int, default=None)
    ruling_query.add_argument("--json", action="store_true")

    rulings = subparsers.add_parser(
        "download-rulings",
        help="download ygocdb ruling API payloads into a local cache",
    )
    rulings.add_argument("--db", type=Path, default=settings.cards_db_path)
    rulings.add_argument(
        "--out",
        type=Path,
        default=settings.ygocdb_cache_path,
        help="cache directory for per-card ygocdb JSON payloads",
    )
    rulings.add_argument("--limit", type=int, default=None)
    rulings.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="delay in seconds between API requests",
    )
    rulings.add_argument(
        "--timeout",
        type=float,
        default=settings.deepseek_timeout_seconds,
        help="per-request timeout in seconds",
    )
    rulings.add_argument("--force", action="store_true", help="redownload existing payloads")
    rulings.add_argument("--fail-fast", action="store_true")
    rulings.add_argument(
        "--use-proxy",
        action="store_true",
        help="use system proxy settings for ygocdb API requests; disabled by default",
    )
    rulings.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress every N cards; use 1 for per-card output",
    )

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
        "--llm-rerank",
        action="store_true",
        help="rerank hybrid candidates using DeepSeek as an LLM judge",
    )
    query.add_argument(
        "--llm",
        action="store_true",
        help="synthesize final answer with DeepSeek through LangChain",
    )

    translate = subparsers.add_parser("translate", help="translate arbitrary text")
    translate.add_argument("text")
    translate.add_argument(
        "--source-lang",
        default="auto",
        help="source language, or auto to let the model infer it",
    )
    translate.add_argument(
        "--target-lang",
        default="zh-CN",
        help="target language; defaults to zh-CN",
    )
    translate.add_argument(
        "--structured-max-block-chars",
        type=int,
        default=None,
        help="optional maximum structured bot message block length",
    )

    adjudicate = subparsers.add_parser("adjudicate", help="answer a ruling scenario")
    adjudicate.add_argument("scenario")
    adjudicate.add_argument("--db", type=Path, default=settings.cards_db_path)
    adjudicate.add_argument(
        "--ygocdb-cache",
        type=Path,
        default=settings.ygocdb_cache_path,
        help="local ygocdb API cache directory or JSON file",
    )
    adjudicate.add_argument(
        "--rules-epub",
        type=Path,
        default=settings.ocg_rules_epub_path,
        help="local OCG rules EPUB path",
    )
    adjudicate.add_argument(
        "--llm",
        action="store_true",
        help="synthesize final adjudication with DeepSeek using gathered evidence",
    )
    adjudicate.add_argument(
        "--simulate",
        action="store_true",
        help="request local simulator verification when available",
    )
    adjudicate.add_argument(
        "--allow-ygocdb-fetch",
        action="store_true",
        help="allow fetching missing ygocdb API payloads from the network",
    )
    adjudicate.add_argument(
        "--agent",
        action="store_true",
        help="use the LangGraph adjudication agent path",
    )
    adjudicate.add_argument(
        "--trace",
        action="store_true",
        help="include the adjudication agent tool trace in JSON output",
    )
    adjudicate.add_argument(
        "--progress",
        action="store_true",
        help="将实时裁定进度输出到 stderr",
    )
    adjudicate.add_argument(
        "--output-format",
        choices=("both", "json", "natural"),
        default="both",
        help="输出格式；默认同时输出自然语言和 JSON",
    )

    assistant = subparsers.add_parser(
        "agent",
        help="让上层 Agent 选择查卡、翻译或裁定工具",
    )
    assistant.add_argument("text")
    assistant.add_argument("--db", type=Path, default=settings.cards_db_path)
    assistant.add_argument("--semantic", action="store_true")
    assistant.add_argument("--rerank", action="store_true")
    assistant.add_argument("--llm-rerank", action="store_true")
    assistant.add_argument("--max-steps", type=int, default=6)
    assistant.add_argument(
        "--ygocdb-cache",
        type=Path,
        default=settings.ygocdb_cache_path,
    )
    assistant.add_argument(
        "--rules-epub",
        type=Path,
        default=settings.ocg_rules_epub_path,
    )
    assistant.add_argument("--trace", action="store_true")
    assistant.add_argument("--progress", action="store_true")
    assistant.add_argument(
        "--output-format",
        choices=("both", "json", "natural"),
        default="both",
    )

    web = subparsers.add_parser("web", help="start local web UI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=7860)
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
        if args.command == "download-rulings":
            return download_rulings(
                args.db,
                args.out,
                limit=args.limit,
                delay_seconds=args.delay,
                timeout_seconds=args.timeout,
                force=args.force,
                fail_fast=args.fail_fast,
                progress_every=args.progress_every,
                use_proxy=args.use_proxy,
            )
        if args.command == "build-index":
            return build_index(
                args.db,
                settings,
                batch_size=args.batch_size,
                limit=args.limit,
                reset=args.reset,
            )
        if args.command == "build-ruling-index":
            return build_ruling_index_command(
                args.db,
                settings,
                ygocdb_cache_path=args.ygocdb_cache,
                rules_epub_path=args.rules_epub,
                faq_only=args.faq_only,
                rules_only=args.rules_only,
                batch_size=args.batch_size,
                limit=args.limit,
                reset=args.reset,
            )
        if args.command == "inspect-ruling-index":
            return inspect_ruling_index_command(settings, json_output=args.json)
        if args.command == "query-rulings":
            return query_rulings_command(
                args.query,
                args.db,
                settings,
                ygocdb_cache_path=args.ygocdb_cache,
                rules_epub_path=args.rules_epub,
                semantic=args.semantic,
                rerank=args.rerank,
                llm_rerank=args.llm_rerank,
                top_k=args.top_k,
                fused_top_k=args.fused_top_k,
                json_output=args.json,
            )
        if args.command == "query":
            return query_cards(
                args.query,
                args.db,
                args.top_k,
                settings,
                semantic=args.semantic,
                rerank=args.rerank,
                llm_rerank=args.llm_rerank,
                use_llm=args.llm,
                rerank_candidates=args.rerank_candidates,
            )
        if args.command == "translate":
            return translate_text(
                args.text,
                settings,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                structured_max_block_chars=args.structured_max_block_chars,
            )
        if args.command == "adjudicate":
            return adjudicate_scenario(
                args.scenario,
                args.db,
                settings,
                ygocdb_cache_path=args.ygocdb_cache,
                rules_epub_path=args.rules_epub,
                use_llm=args.llm,
                simulate=args.simulate,
                allow_ygocdb_fetch=args.allow_ygocdb_fetch,
                use_agent=args.agent,
                include_trace=args.trace,
                show_progress=args.progress,
                output_format=args.output_format,
            )
        if args.command == "agent":
            return run_assistant_agent(
                args.text,
                args.db,
                settings,
                semantic=args.semantic,
                rerank=args.rerank,
                llm_rerank=args.llm_rerank,
                max_steps=args.max_steps,
                ygocdb_cache_path=args.ygocdb_cache,
                rules_epub_path=args.rules_epub,
                include_trace=args.trace,
                show_progress=args.progress,
                output_format=args.output_format,
            )
        if args.command == "web":
            return run_web(args.host, args.port)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
