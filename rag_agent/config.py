from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cards_db_path: Path
    chroma_persist_dir: Path
    embedding_model: str
    reranker_model: str
    device: str
    embedding_device: str
    reranker_device: str
    reranker_device_explicit: bool
    rerank_provider: str
    llm_rerank_max_candidates: int
    deepseek_timeout_seconds: float
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        data_dir = Path(values.get("RAG_DATA_DIR", "data"))
        device = values.get("RAG_DEVICE", "auto")
        return cls(
            data_dir=data_dir,
            cards_db_path=Path(values.get("RAG_CARDS_DB", str(data_dir / "cards.cdb"))),
            chroma_persist_dir=Path(
                values.get("CHROMA_PERSIST_DIR", str(data_dir / "chroma"))
            ),
            embedding_model=values.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=values.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            device=device,
            embedding_device=values.get("RAG_EMBEDDING_DEVICE", device),
            reranker_device=values.get("RAG_RERANKER_DEVICE", device),
            reranker_device_explicit="RAG_RERANKER_DEVICE" in values,
            rerank_provider=values.get("RAG_RERANK_PROVIDER", "none"),
            llm_rerank_max_candidates=int(
                values.get("RAG_LLM_RERANK_MAX_CANDIDATES", "20")
            ),
            deepseek_timeout_seconds=float(
                values.get("DEEPSEEK_TIMEOUT_SECONDS", "60")
            ),
            deepseek_api_key=values.get("DEEPSEEK_API_KEY"),
            deepseek_base_url=values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=values.get("DEEPSEEK_MODEL", "deepseek-chat"),
        )
