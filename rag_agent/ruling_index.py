from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from rag_agent.adjudication import (
    EvidenceItem,
    RuleIndex,
    RuleSection,
    YgocdbCache,
    YgocdbFaqDocument,
    YgocdbRulingIndex,
    YgocdbSupplementDocument,
)
from rag_agent.card_metadata import decode_effect_category_labels
from rag_agent.cards import Card, RetrievalDocument, normalize_effect_text
from rag_agent.retrieval.reranker import BgeReranker, SubprocessReranker
from rag_agent.retrieval.sparse import tokenize_text


RULING_INDEX_SCHEMA_VERSION = 1
RulingDocumentKind = Literal["faq", "card_supplement", "rule_passage"]


@dataclass(frozen=True)
class RulingDocument:
    evidence_id: str
    kind: RulingDocumentKind
    source: str
    title: str
    text: str
    card_ids: tuple[int, ...] = ()
    primary_card_ids: tuple[int, ...] = ()
    refer_card_ids: tuple[int, ...] = ()
    relationship_hints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return normalize_effect_text(
            "\n".join(
                part
                for part in [
                    self.evidence_id,
                    self.kind,
                    self.source,
                    self.title,
                    " ".join(self.relationship_hints),
                    self.text,
                    _metadata_search_text(self.metadata),
                ]
                if part
            )
        )

    def to_retrieval_document(self) -> RetrievalDocument:
        metadata = {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source": self.source,
            "title": self.title,
            "card_ids_json": json.dumps(list(self.card_ids), ensure_ascii=False),
            "primary_card_ids_json": json.dumps(list(self.primary_card_ids), ensure_ascii=False),
            "refer_card_ids_json": json.dumps(list(self.refer_card_ids), ensure_ascii=False),
            "relationship_hints_json": json.dumps(list(self.relationship_hints), ensure_ascii=False),
        } | _json_safe_metadata(self.metadata)
        return RetrievalDocument(page_content=self.searchable_text, metadata=metadata)

    def to_evidence(self, relationship: str | None = None) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.evidence_id,
            kind=self.kind,
            source=self.source,
            title=self.title,
            text=self.text,
            relationship=relationship or _default_relationship(self.kind),
            card_ids=self.card_ids,
            metadata=dict(self.metadata)
            | {
                "primary_card_ids": list(self.primary_card_ids),
                "refer_card_ids": list(self.refer_card_ids),
                "relationship_hints": list(self.relationship_hints),
            },
        )


@dataclass(frozen=True)
class RulingCandidate:
    evidence_id: str
    score: float
    source: str
    document: RulingDocument
    relationship: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_evidence(self) -> EvidenceItem:
        return self.document.to_evidence(self.relationship)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "score": self.score,
            "source": self.source,
            "relationship": self.relationship,
            "kind": self.document.kind,
            "title": self.document.title,
            "snippet": self.document.text[:240],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RulingIndexManifest:
    schema_version: int
    index_kind: str
    persist_dir: str
    collection_name: str
    source_paths: tuple[str, ...]
    source_count: int
    document_count: int
    embedding_model: str
    built_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "index_kind": self.index_kind,
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "source_paths": list(self.source_paths),
            "source_count": self.source_count,
            "document_count": self.document_count,
            "embedding_model": self.embedding_model,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RulingIndexManifest":
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            index_kind=str(payload.get("index_kind") or ""),
            persist_dir=str(payload.get("persist_dir") or ""),
            collection_name=str(payload.get("collection_name") or ""),
            source_paths=tuple(str(item) for item in payload.get("source_paths") or ()),
            source_count=int(payload.get("source_count") or 0),
            document_count=int(payload.get("document_count") or 0),
            embedding_model=str(payload.get("embedding_model") or ""),
            built_at=str(payload.get("built_at") or ""),
        )


@dataclass(frozen=True)
class ManifestStatus:
    usable: bool
    warnings: tuple[str, ...] = ()
    manifest: RulingIndexManifest | None = None


@dataclass(frozen=True)
class RulingIndexBuildResult:
    index_kind: str
    source_count: int
    document_count: int
    collection_name: str
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_kind": self.index_kind,
            "source_count": self.source_count,
            "document_count": self.document_count,
            "collection_name": self.collection_name,
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True)
class RulingIndexInspection:
    index_kind: str
    manifest_path: Path
    manifest_status: ManifestStatus
    collection_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_kind": self.index_kind,
            "manifest_path": str(self.manifest_path),
            "usable": self.manifest_status.usable,
            "warnings": list(self.manifest_status.warnings),
            "manifest": (
                self.manifest_status.manifest.to_dict()
                if self.manifest_status.manifest
                else None
            ),
            "collection_count": self.collection_count,
        }


@dataclass(frozen=True)
class RulingRetrievalResult:
    candidates: tuple[RulingCandidate, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
        }


def build_ygocdb_ruling_documents(
    ruling_index: YgocdbRulingIndex,
    cards_by_id: Mapping[int, Card] | None = None,
) -> list[RulingDocument]:
    cards_by_id = cards_by_id or {}
    documents: list[RulingDocument] = []
    for faq in ruling_index.faq_by_fid.values():
        documents.append(faq_to_ruling_document(faq, cards_by_id))
    for supplement in ruling_index.supplement_by_card_id.values():
        documents.append(supplement_to_ruling_document(supplement, cards_by_id))
    documents.sort(key=lambda document: document.evidence_id)
    return documents


def faq_to_ruling_document(
    faq: YgocdbFaqDocument,
    cards_by_id: Mapping[int, Card] | None = None,
) -> RulingDocument:
    cards_by_id = cards_by_id or {}
    card_ids = tuple(sorted(set(faq.primary_card_ids).union(faq.refer_card_ids)))
    metadata = _card_metadata(card_ids, cards_by_id) | {
        "fid": faq.fid,
        "date": faq.date,
        "source_url": faq.source_url,
        "snapshot_at": faq.snapshot_at,
        "html_refer_card_ids": list(faq.html_refer_card_ids),
    }
    return RulingDocument(
        evidence_id=faq.ruling_id,
        kind="faq",
        source="ygocdb",
        title=faq.title_plain,
        text=faq.searchable_text,
        card_ids=card_ids,
        primary_card_ids=faq.primary_card_ids,
        refer_card_ids=faq.refer_card_ids,
        relationship_hints=("direct_card", "related_card", "analogous"),
        metadata=metadata,
    )


def supplement_to_ruling_document(
    supplement: YgocdbSupplementDocument,
    cards_by_id: Mapping[int, Card] | None = None,
) -> RulingDocument:
    cards_by_id = cards_by_id or {}
    card_ids = tuple(sorted({supplement.primary_card_id}.union(supplement.refer_card_ids)))
    metadata = _card_metadata(card_ids, cards_by_id) | {
        "date": supplement.date,
        "source_url": supplement.source_url,
        "snapshot_at": supplement.snapshot_at,
        "html_refer_card_ids": list(supplement.html_refer_card_ids),
    }
    return RulingDocument(
        evidence_id=supplement.supplement_id,
        kind="card_supplement",
        source="ygocdb",
        title=f"ygocdb supplement: {supplement.primary_card_id}",
        text=supplement.text_plain,
        card_ids=card_ids,
        primary_card_ids=(supplement.primary_card_id,),
        refer_card_ids=supplement.refer_card_ids,
        relationship_hints=("card_supplement",),
        metadata=metadata,
    )


def build_rule_ruling_documents(rule_index: RuleIndex) -> list[RulingDocument]:
    return [rule_section_to_ruling_document(section) for section in rule_index.sections]


def rule_section_to_ruling_document(section: RuleSection) -> RulingDocument:
    return RulingDocument(
        evidence_id=section.rule_id,
        kind="rule_passage",
        source="ocgrules",
        title=section.title,
        text=section.text,
        relationship_hints=("rule_document",),
        metadata={
            "chapter_path": list(section.chapter_path),
            "chapter_path_text": " > ".join(section.chapter_path),
            "chapter_id": section.chapter_id,
            "xhtml_path": section.xhtml_path,
            "anchor": section.anchor,
        },
    )


def load_ygocdb_ruling_documents_from_cache(
    cache_path: Path,
    cards: Sequence[Card],
    *,
    limit: int | None = None,
) -> tuple[list[RulingDocument], int]:
    cards_by_id = {card.card_id: card for card in cards}
    cache = YgocdbCache(cache_path)
    payloads: list[Mapping[str, Any]] = []
    source_count = 0
    for card in cards[:limit] if limit is not None else cards:
        payload = cache.load_card_payload(card.card_id)
        if payload is None:
            continue
        payloads.append(payload)
        source_count += 1
    if not payloads:
        return [], source_count
    index = YgocdbRulingIndex.from_payloads(payloads)
    return build_ygocdb_ruling_documents(index, cards_by_id), source_count


def load_rule_documents_from_epub(
    epub_path: Path,
    *,
    limit: int | None = None,
) -> tuple[list[RulingDocument], int]:
    if not epub_path.exists():
        return [], 0
    rule_index = RuleIndex.from_epub(epub_path)
    documents = build_rule_ruling_documents(rule_index)
    if limit is not None:
        documents = documents[:limit]
    return documents, len(rule_index.sections)


def build_persistent_ruling_index(
    documents: Sequence[RulingDocument],
    *,
    index_kind: str,
    persist_dir: Path,
    collection_name: str,
    manifest_path: Path,
    source_paths: Sequence[Path],
    source_count: int,
    embedding_model: str,
    embedding_function: object,
    batch_size: int = 128,
    reset: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RulingIndexBuildResult:
    store = RulingVectorStore(persist_dir, collection_name, embedding_function)
    store.build(
        documents,
        batch_size=batch_size,
        reset=reset,
        progress_callback=progress_callback,
    )
    manifest = create_manifest(
        index_kind=index_kind,
        persist_dir=persist_dir,
        collection_name=collection_name,
        source_paths=source_paths,
        source_count=source_count,
        document_count=len(documents),
        embedding_model=embedding_model,
    )
    write_manifest(manifest_path, manifest)
    return RulingIndexBuildResult(
        index_kind=index_kind,
        source_count=source_count,
        document_count=len(documents),
        collection_name=collection_name,
        manifest_path=manifest_path,
    )


def inspect_persistent_ruling_index(
    *,
    index_kind: str,
    persist_dir: Path,
    collection_name: str,
    manifest_path: Path,
    embedding_model: str,
    embedding_function: object,
) -> RulingIndexInspection:
    status = validate_manifest(
        manifest_path,
        index_kind=index_kind,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    collection_count: int | None = None
    if status.manifest is not None:
        try:
            collection_count = RulingVectorStore(
                persist_dir,
                collection_name,
                embedding_function,
            ).count()
        except Exception:
            collection_count = None
    return RulingIndexInspection(
        index_kind=index_kind,
        manifest_path=manifest_path,
        manifest_status=status,
        collection_count=collection_count,
    )


class SparseRulingRetriever:
    def __init__(self, documents: Sequence[RulingDocument]) -> None:
        self.documents = list(documents)
        self.doc_freq: Counter[str] = Counter()
        self.term_counts_by_id: dict[str, Counter[str]] = {}
        self.document_by_id = {document.evidence_id: document for document in self.documents}
        for document in self.documents:
            counts = Counter(tokenize_text(document.searchable_text))
            self.term_counts_by_id[document.evidence_id] = counts
            self.doc_freq.update(counts.keys())

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        card_ids: Sequence[int] = (),
        chapter_ids: Sequence[str] = (),
        relationship: str = "analogous",
    ) -> list[RulingCandidate]:
        query_terms = Counter(tokenize_text(query))
        if not query_terms and not card_ids:
            return []
        requested_cards = set(card_ids)
        allowed_chapters = set(chapter_ids)
        total_docs = max(len(self.documents), 1)
        candidates: list[RulingCandidate] = []
        for document in self.documents:
            if allowed_chapters and str(document.metadata.get("xhtml_path") or "") not in allowed_chapters:
                continue
            score = 0.0
            counts = self.term_counts_by_id[document.evidence_id]
            for term, query_count in query_terms.items():
                term_count = counts.get(term, 0)
                if not term_count:
                    continue
                idf = math.log((1 + total_docs) / (1 + self.doc_freq[term])) + 1
                score += query_count * term_count * idf
            relationship_label = relationship_for_document(document, requested_cards, relationship)
            if requested_cards:
                if requested_cards.intersection(document.primary_card_ids):
                    score += 1000.0
                elif requested_cards.intersection(document.refer_card_ids):
                    score += 500.0
            if score > 0:
                candidates.append(
                    RulingCandidate(
                        evidence_id=document.evidence_id,
                        score=score,
                        source="sparse",
                        document=document,
                        relationship=relationship_label,
                    )
                )
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.evidence_id))
        return candidates[:top_k]


class RulingVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_function: object,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function

    def _load_chroma(self):
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise RuntimeError(
                "langchain-chroma is required for persistent ruling indexes. "
                "Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc
        return Chroma

    def build(
        self,
        documents: Sequence[RulingDocument],
        *,
        batch_size: int = 128,
        reset: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Any:
        Chroma = self._load_chroma()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding_function,
        )
        if reset:
            try:
                store.delete_collection()
            except Exception:
                pass
            store = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedding_function,
            )
        total = len(documents)
        for start in range(0, total, batch_size):
            batch = documents[start : start + batch_size]
            store.add_texts(
                texts=[document.searchable_text for document in batch],
                metadatas=[_clean_chroma_metadata(document.to_retrieval_document().metadata) for document in batch],
                ids=[document.evidence_id for document in batch],
            )
            if progress_callback is not None:
                progress_callback(min(start + len(batch), total), total)
        return store

    def load(self):
        Chroma = self._load_chroma()
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding_function,
        )

    def search(
        self,
        query: str,
        documents_by_id: Mapping[str, RulingDocument],
        *,
        top_k: int = 10,
        card_ids: Sequence[int] = (),
        chapter_ids: Sequence[str] = (),
        relationship: str = "analogous",
    ) -> list[RulingCandidate]:
        store = self.load()
        metadata_filter: dict[str, Any] | None = None
        if len(chapter_ids) == 1:
            metadata_filter = {"xhtml_path": str(chapter_ids[0])}
        elif chapter_ids:
            metadata_filter = {"xhtml_path": {"$in": [str(item) for item in chapter_ids]}}
        kwargs = {"filter": metadata_filter} if metadata_filter else {}
        results = store.similarity_search_with_score(query, k=top_k, **kwargs)
        requested_cards = set(card_ids)
        candidates: list[RulingCandidate] = []
        for document, distance in results:
            metadata = dict(document.metadata)
            evidence_id = str(metadata.get("evidence_id") or "")
            ruling_document = documents_by_id.get(evidence_id) or ruling_document_from_metadata(
                metadata,
                document.page_content,
            )
            candidates.append(
                RulingCandidate(
                    evidence_id=evidence_id,
                    score=-float(distance),
                    source="dense",
                    document=ruling_document,
                    relationship=relationship_for_document(ruling_document, requested_cards, relationship),
                    metadata=metadata,
                )
            )
        return candidates

    def count(self) -> int:
        store = self.load()
        try:
            return int(store._collection.count())
        except Exception as exc:
            raise RuntimeError("Unable to read ruling Chroma collection count.") from exc


class PersistentRulingRetriever:
    def __init__(
        self,
        documents: Sequence[RulingDocument],
        *,
        dense_store: RulingVectorStore | None = None,
        manifest_status: ManifestStatus | None = None,
        local_reranker: BgeReranker | SubprocessReranker | None = None,
        llm: object | None = None,
        llm_max_candidates: int = 20,
        warning_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.documents = list(documents)
        self.documents_by_id = {document.evidence_id: document for document in self.documents}
        self.sparse = SparseRulingRetriever(self.documents)
        self.dense_store = dense_store
        self.manifest_status = manifest_status
        self.local_reranker = local_reranker
        self.llm = llm
        self.llm_max_candidates = llm_max_candidates
        self.warning_callback = warning_callback

    def search(
        self,
        query: str,
        *,
        card_ids: Sequence[int] = (),
        chapter_ids: Sequence[str] = (),
        relationship: str = "analogous",
        sparse_top_k: int = 20,
        dense_top_k: int = 20,
        fused_top_k: int = 20,
        final_top_k: int = 10,
        semantic: bool = True,
        rerank_provider: str = "none",
    ) -> RulingRetrievalResult:
        warnings: list[str] = []
        sparse_candidates = self.sparse.search(
            query,
            top_k=sparse_top_k,
            card_ids=card_ids,
            chapter_ids=chapter_ids,
            relationship=relationship,
        )
        dense_candidates: list[RulingCandidate] = []
        if semantic and self.dense_store is not None:
            if self.manifest_status is not None and not self.manifest_status.usable:
                warnings.extend(self.manifest_status.warnings)
            else:
                try:
                    dense_kwargs: dict[str, Any] = {
                        "top_k": dense_top_k,
                        "card_ids": card_ids,
                        "relationship": relationship,
                    }
                    if chapter_ids:
                        dense_kwargs["chapter_ids"] = chapter_ids
                    dense_candidates = self.dense_store.search(
                        query,
                        self.documents_by_id,
                        **dense_kwargs,
                    )
                except Exception as exc:
                    warnings.append(f"Persistent ruling dense retrieval failed: {exc}")
        candidates = hybrid_ruling_retrieval(
            query,
            sparse_candidates=sparse_candidates,
            dense_candidates=dense_candidates,
            top_k=fused_top_k,
        )
        provider = normalize_ruling_rerank_provider(rerank_provider)
        if provider == "local" and self.local_reranker is not None:
            try:
                candidates = rerank_ruling_candidates_locally(
                    query,
                    candidates,
                    self.local_reranker,
                )
            except Exception as exc:
                warnings.append(f"Ruling local rerank failed; using fused order. {exc}")
        elif provider == "llm" and self.llm is not None:
            candidates = rerank_ruling_candidates_with_llm(
                query,
                candidates,
                llm=self.llm,
                max_candidates=self.llm_max_candidates,
                warning_callback=warnings.append,
            )
        elif provider in {"local", "llm"}:
            warnings.append(f"Ruling rerank provider '{provider}' was requested but is not configured.")
        for warning in warnings:
            if self.warning_callback is not None:
                self.warning_callback(warning)
        return RulingRetrievalResult(
            candidates=tuple(candidates[:final_top_k]),
            warnings=tuple(warnings),
        )


def normalize_ruling_rerank_provider(provider: str | None) -> str:
    normalized = (provider or "none").strip().lower()
    aliases = {"": "none", "false": "none", "off": "none", "no": "none", "bge": "local", "local-bge": "local", "llm-rerank": "llm"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "local", "llm"}:
        raise ValueError("ruling rerank provider must be one of: none, local, llm.")
    return normalized


def ruling_document_from_metadata(metadata: Mapping[str, Any], text: str) -> RulingDocument:
    return RulingDocument(
        evidence_id=str(metadata.get("evidence_id") or ""),
        kind=str(metadata.get("kind") or "faq"),  # type: ignore[arg-type]
        source=str(metadata.get("source") or ""),
        title=str(metadata.get("title") or ""),
        text=text,
        card_ids=_loads_int_tuple(metadata.get("card_ids_json")),
        primary_card_ids=_loads_int_tuple(metadata.get("primary_card_ids_json")),
        refer_card_ids=_loads_int_tuple(metadata.get("refer_card_ids_json")),
        relationship_hints=tuple(str(item) for item in _loads_json_list(metadata.get("relationship_hints_json"))),
        metadata=dict(metadata),
    )


def hybrid_ruling_retrieval(
    query: str,
    *,
    sparse_candidates: Sequence[RulingCandidate] = (),
    dense_candidates: Sequence[RulingCandidate] = (),
    top_k: int = 10,
) -> list[RulingCandidate]:
    scores: dict[str, float] = {}
    tie_breakers: dict[str, float] = {}
    by_id: dict[str, RulingCandidate] = {}
    for ranked_list in (sparse_candidates, dense_candidates):
        for rank, candidate in enumerate(ranked_list, start=1):
            scores[candidate.evidence_id] = scores.get(candidate.evidence_id, 0.0) + 1.0 / (60 + rank)
            tie_breakers[candidate.evidence_id] = max(
                tie_breakers.get(candidate.evidence_id, float("-inf")),
                math.log1p(max(candidate.score, 0.0)),
            )
            by_id.setdefault(candidate.evidence_id, candidate)
    fused = [
        RulingCandidate(
            evidence_id=evidence_id,
            score=score,
            source="hybrid",
            document=by_id[evidence_id].document,
            relationship=by_id[evidence_id].relationship,
            metadata=dict(by_id[evidence_id].metadata),
        )
        for evidence_id, score in scores.items()
    ]
    fused.sort(
        key=lambda candidate: (
            -candidate.score,
            -tie_breakers.get(candidate.evidence_id, 0.0),
            candidate.evidence_id,
        )
    )
    return fused[:top_k]


def rerank_ruling_candidates_locally(
    query: str,
    candidates: Sequence[RulingCandidate],
    reranker: BgeReranker | SubprocessReranker,
) -> list[RulingCandidate]:
    from rag_agent.retrieval.hybrid import Candidate

    id_by_number = {index + 1: candidate.evidence_id for index, candidate in enumerate(candidates)}
    candidate_by_id = {candidate.evidence_id: candidate for candidate in candidates}
    card_candidates = [
        Candidate(
            card_id=index,
            score=candidate.score,
            source=candidate.source,
            metadata={
                "name": candidate.document.title,
                "desc": candidate.document.text,
                "text": candidate.document.text,
            },
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    reranked = reranker.rerank(query, card_candidates)
    results: list[RulingCandidate] = []
    for item in reranked:
        evidence_id = id_by_number[item.card_id]
        original = candidate_by_id[evidence_id]
        results.append(
            RulingCandidate(
                evidence_id=evidence_id,
                score=item.score,
                source="reranker",
                document=original.document,
                relationship=original.relationship,
                metadata=dict(original.metadata),
            )
        )
    return results


@dataclass(frozen=True)
class RulingLlmJudgeScore:
    evidence_id: str
    score: float
    reason: str


def build_ruling_llm_rerank_prompt(
    query: str,
    candidates: Sequence[RulingCandidate],
) -> str:
    payload = [
        {
            "evidence_id": candidate.evidence_id,
            "kind": candidate.document.kind,
            "relationship": candidate.relationship,
            "title": candidate.document.title,
            "retrieval_score": f"{candidate.score:.4f}",
            "snippet": candidate.document.text[:1200],
        }
        for candidate in candidates
    ]
    return "\n".join(
        [
            "你是游戏王 OCG 裁定证据重排评审。",
            "只能给候选列表中已有的 evidence_id 排序，不要引入外部证据。",
            "当 direct_card 和 related_card FAQ 能回答场景时优先使用；analogous FAQ 和规则文档用于补充机制覆盖。",
            "只返回严格 JSON：{\"results\":[{\"evidence_id\":\"...\",\"score\":0.0,\"reason\":\"简短理由\"}]}",
            "",
            "查询：",
            query,
            "",
            "候选证据：",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def parse_ruling_llm_rerank_response(
    content: object,
    *,
    allowed_ids: set[str],
) -> list[RulingLlmJudgeScore]:
    text = str(getattr(content, "content", content)).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise RuntimeError("Unable to parse ruling LLM rerank response as JSON.") from exc
    raw_results = payload if isinstance(payload, list) else payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raise RuntimeError("Unable to parse ruling LLM rerank response: expected results list.")
    parsed: list[RulingLlmJudgeScore] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id not in allowed_ids or evidence_id in seen:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        reason = normalize_effect_text(str(item.get("reason") or ""))
        parsed.append(RulingLlmJudgeScore(evidence_id=evidence_id, score=score, reason=reason))
        seen.add(evidence_id)
    return parsed


def rerank_ruling_candidates_with_llm(
    query: str,
    candidates: Sequence[RulingCandidate],
    *,
    llm: object,
    max_candidates: int,
    warning_callback: Callable[[str], None] | None = None,
) -> list[RulingCandidate]:
    if not candidates:
        return []
    limited = list(candidates[: max(max_candidates, 0)])
    overflow = list(candidates[len(limited) :])
    if not limited:
        return list(candidates)
    try:
        response = llm.invoke(build_ruling_llm_rerank_prompt(query, limited))
        scores = parse_ruling_llm_rerank_response(
            response,
            allowed_ids={candidate.evidence_id for candidate in limited},
        )
    except Exception as exc:
        if warning_callback is not None:
            warning_callback(
                "Ruling LLM rerank failed; falling back to previous evidence order. "
                f"{exc}"
            )
        return list(candidates)
    score_by_id = {score.evidence_id: score for score in scores}
    original_position = {candidate.evidence_id: index for index, candidate in enumerate(candidates)}
    scored: list[RulingCandidate] = []
    unscored: list[RulingCandidate] = []
    for candidate in limited:
        score = score_by_id.get(candidate.evidence_id)
        if score is None:
            unscored.append(
                _with_ruling_llm_reason(candidate, candidate.score, "LLM judge did not score this candidate.")
            )
            continue
        scored.append(_with_ruling_llm_reason(candidate, score.score, score.reason))
    scored.sort(key=lambda candidate: (-candidate.score, candidate.evidence_id))
    unscored.extend(
        _with_ruling_llm_reason(candidate, candidate.score, "Candidate was not sent to LLM judge due to max candidate limit.")
        for candidate in overflow
    )
    unscored.sort(key=lambda candidate: original_position[candidate.evidence_id])
    return scored + unscored


def write_manifest(path: Path, manifest: RulingIndexManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(path: Path) -> RulingIndexManifest | None:
    if not path.exists():
        return None
    return RulingIndexManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def validate_manifest(
    path: Path,
    *,
    index_kind: str,
    collection_name: str,
    embedding_model: str,
) -> ManifestStatus:
    manifest = read_manifest(path)
    if manifest is None:
        return ManifestStatus(False, (f"Ruling index manifest is missing: {path}",), None)
    warnings: list[str] = []
    if manifest.schema_version != RULING_INDEX_SCHEMA_VERSION:
        warnings.append("Ruling index manifest schema version is incompatible.")
    if manifest.index_kind != index_kind:
        warnings.append("Ruling index manifest kind does not match requested index.")
    if manifest.collection_name != collection_name:
        warnings.append("Ruling index manifest collection name does not match settings.")
    if manifest.embedding_model != embedding_model:
        warnings.append("Ruling index manifest embedding model does not match settings.")
    if manifest.document_count <= 0:
        warnings.append("Ruling index manifest has no documents.")
    return ManifestStatus(not warnings, tuple(warnings), manifest)


def create_manifest(
    *,
    index_kind: str,
    persist_dir: Path,
    collection_name: str,
    source_paths: Sequence[Path],
    source_count: int,
    document_count: int,
    embedding_model: str,
) -> RulingIndexManifest:
    return RulingIndexManifest(
        schema_version=RULING_INDEX_SCHEMA_VERSION,
        index_kind=index_kind,
        persist_dir=str(persist_dir),
        collection_name=collection_name,
        source_paths=tuple(str(path) for path in source_paths),
        source_count=source_count,
        document_count=document_count,
        embedding_model=embedding_model,
        built_at=datetime.now(timezone.utc).isoformat(),
    )


def relationship_for_document(
    document: RulingDocument,
    requested_cards: set[int],
    default_relationship: str,
) -> str:
    if document.kind == "card_supplement":
        return "card_supplement"
    if document.kind == "rule_passage":
        return "rule_document"
    labels: list[str] = []
    if requested_cards.intersection(document.primary_card_ids):
        labels.append("direct_card")
    if requested_cards.intersection(document.refer_card_ids):
        labels.append("related_card")
    if not labels:
        labels.append(default_relationship)
    return ",".join(dict.fromkeys(labels))


def _with_ruling_llm_reason(
    candidate: RulingCandidate,
    score: float,
    reason: str,
) -> RulingCandidate:
    return RulingCandidate(
        evidence_id=candidate.evidence_id,
        score=float(score),
        source="llm_reranker",
        document=candidate.document,
        relationship=candidate.relationship,
        metadata=dict(candidate.metadata) | {"llm_judge_reason": reason},
    )


def _card_metadata(card_ids: Sequence[int], cards_by_id: Mapping[int, Card]) -> dict[str, Any]:
    card_names: list[str] = []
    effect_tags: list[str] = []
    for card_id in card_ids:
        card = cards_by_id.get(card_id)
        if card is None:
            continue
        card_names.append(card.name)
        effect_tags.extend(decode_effect_category_labels(card.category))
    return {
        "card_names": card_names,
        "effect_tags": list(dict.fromkeys(effect_tags)),
        "card_names_text": " ".join(card_names),
        "effect_tags_text": " ".join(dict.fromkeys(effect_tags)),
    }


def _metadata_search_text(metadata: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("card_names_text", "effect_tags_text", "chapter_path_text"):
        value = metadata.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, (list, tuple, dict)):
            safe[f"{key}_json"] = json.dumps(value, ensure_ascii=False)
    return safe


def _clean_chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def _loads_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        loaded = json.loads(str(value))
    except Exception:
        return []
    return loaded if isinstance(loaded, list) else []


def _loads_int_tuple(value: Any) -> tuple[int, ...]:
    ints: list[int] = []
    for item in _loads_json_list(value):
        try:
            ints.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(ints)


def _default_relationship(kind: str) -> str:
    if kind == "card_supplement":
        return "card_supplement"
    if kind == "rule_passage":
        return "rule_document"
    return "analogous"
