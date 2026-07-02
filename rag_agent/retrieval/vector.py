from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rag_agent.cards import RetrievalDocument
from rag_agent.retrieval.hybrid import Candidate
from rag_agent.structured_query import StructuredQueryFilters


@dataclass(frozen=True)
class ChromaConfig:
    persist_dir: Path
    collection_name: str = "ygopro_cards_zh_cn"


class ChromaVectorStore:
    def __init__(self, config: ChromaConfig, embedding_function: object) -> None:
        self.config = config
        self.embedding_function = embedding_function

    def _load_chroma(self):
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise RuntimeError(
                "langchain-chroma is required for Chroma vector storage. "
                "Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc
        return Chroma

    def build(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        batch_size: int = 128,
        reset: bool = False,
        progress_callback=None,
    ):
        Chroma = self._load_chroma()
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_dir),
            embedding_function=self.embedding_function,
        )
        if reset:
            try:
                store.delete_collection()
            except Exception:
                pass
            store = Chroma(
                collection_name=self.config.collection_name,
                persist_directory=str(self.config.persist_dir),
                embedding_function=self.embedding_function,
            )

        total = len(documents)
        for start in range(0, total, batch_size):
            batch = documents[start : start + batch_size]
            store.add_texts(
                texts=[document.page_content for document in batch],
                metadatas=[_clean_metadata(document.metadata) for document in batch],
                ids=[str(document.metadata["card_id"]) for document in batch],
            )
            if progress_callback is not None:
                progress_callback(min(start + len(batch), total), total)
        return store

    def load(self):
        Chroma = self._load_chroma()
        return Chroma(
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_dir),
            embedding_function=self.embedding_function,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        store = self.load()
        kwargs = {"filter": metadata_filter} if metadata_filter else {}
        results = store.similarity_search_with_score(query, k=top_k, **kwargs)
        candidates: list[Candidate] = []
        for document, distance in results:
            metadata = dict(document.metadata)
            card_id = int(metadata["card_id"])
            candidates.append(
                Candidate(
                    card_id=card_id,
                    score=-float(distance),
                    source="dense",
                    metadata=metadata | {"text": document.page_content},
                )
            )
        return candidates

    def count(self) -> int:
        store = self.load()
        try:
            return int(store._collection.count())
        except Exception as exc:
            raise RuntimeError("Unable to read Chroma collection count.") from exc

    @classmethod
    def from_persisted(cls, config: ChromaConfig, embedding_function: object) -> "ChromaVectorStore":
        return cls(config, embedding_function)


def _clean_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def structured_filters_to_chroma_where(
    filters: StructuredQueryFilters,
) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []
    if filters.card_kind:
        conditions.append({"card_kind": {"$eq": filters.card_kind}})
    if filters.attribute:
        conditions.append({"attribute_name": {"$eq": filters.attribute}})
    if filters.race:
        conditions.append({"race_name": {"$eq": filters.race}})
    if filters.level is not None:
        conditions.append({"decoded_level": {"$eq": filters.level}})
    if filters.rank is not None:
        conditions.append({"rank": {"$eq": filters.rank}})
    if filters.link_rating is not None:
        conditions.append({"link_rating": {"$eq": filters.link_rating}})
    for monster_type in filters.monster_types:
        conditions.append({f"is_{monster_type}": {"$eq": True}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
