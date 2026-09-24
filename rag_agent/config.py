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
    ygocdb_cache_path: Path
    ocg_rules_epub_path: Path
    ocg_rules_index_path: Path
    adjudication_ruling_top_k: int
    adjudication_rules_top_k: int
    adjudication_agent_max_iterations: int
    adjudication_agent_total_turn_limit: int
    adjudication_agent_retrieval_limit: int
    adjudication_agent_critique_limit: int
    adjudication_agent_submit_limit: int
    adjudication_card_confidence_threshold: float
    adjudication_card_ambiguity_margin: float
    adjudication_card_resolution_top_k: int
    adjudication_card_resolution_retry_limit: int
    adjudication_llm_scenario_parsing: bool
    adjudication_llm_card_extraction: bool
    adjudication_llm_card_selection: bool
    adjudication_trace_default: bool
    ruling_faq_index_path: Path
    ruling_rules_index_path: Path
    ruling_faq_collection_name: str
    ruling_rules_collection_name: str
    ruling_faq_manifest_path: Path
    ruling_rules_manifest_path: Path
    ruling_sparse_top_k: int
    ruling_dense_top_k: int
    ruling_fused_top_k: int
    ruling_final_top_k: int
    ruling_rerank_provider: str
    riftbound_rag_base_url: str
    riftbound_rag_timeout_seconds: float

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
            ygocdb_cache_path=Path(
                values.get("YGOCDB_CACHE_PATH", str(data_dir / "ygocdb"))
            ),
            ocg_rules_epub_path=Path(
                values.get(
                    "OCG_RULES_EPUB",
                    "ocg-rule-readthedocs-io-zh-cn-latest.epub",
                )
            ),
            ocg_rules_index_path=Path(
                values.get("OCG_RULES_INDEX_PATH", str(data_dir / "ocg_rules"))
            ),
            adjudication_ruling_top_k=int(
                values.get("RAG_ADJUDICATION_RULING_TOP_K", "5")
            ),
            adjudication_rules_top_k=int(
                values.get("RAG_ADJUDICATION_RULES_TOP_K", "5")
            ),
            adjudication_agent_max_iterations=int(
                values.get("RAG_ADJUDICATION_AGENT_MAX_ITERATIONS", "8")
            ),
            adjudication_agent_total_turn_limit=int(
                values.get(
                    "RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT",
                    "14",
                )
            ),
            adjudication_agent_retrieval_limit=int(
                values.get("RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT", "8")
            ),
            adjudication_agent_critique_limit=int(
                values.get("RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT", "5")
            ),
            adjudication_agent_submit_limit=int(
                values.get("RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT", "2")
            ),
            adjudication_card_confidence_threshold=float(
                values.get("RAG_ADJUDICATION_CARD_CONFIDENCE_THRESHOLD", "0.75")
            ),
            adjudication_card_ambiguity_margin=float(
                values.get("RAG_ADJUDICATION_CARD_AMBIGUITY_MARGIN", "0.08")
            ),
            adjudication_card_resolution_top_k=int(
                values.get("RAG_ADJUDICATION_CARD_RESOLUTION_TOP_K", "5")
            ),
            adjudication_card_resolution_retry_limit=int(
                values.get("RAG_ADJUDICATION_CARD_RESOLUTION_RETRY_LIMIT", "1")
            ),
            adjudication_llm_scenario_parsing=values.get(
                "RAG_ADJUDICATION_LLM_SCENARIO_PARSING", "1"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            adjudication_llm_card_extraction=values.get(
                "RAG_ADJUDICATION_LLM_CARD_EXTRACTION", "1"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            adjudication_llm_card_selection=values.get(
                "RAG_ADJUDICATION_LLM_CARD_SELECTION", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            adjudication_trace_default=values.get(
                "RAG_ADJUDICATION_TRACE", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            ruling_faq_index_path=Path(
                values.get("RAG_RULING_FAQ_INDEX_PATH", str(data_dir / "ruling_faq_chroma"))
            ),
            ruling_rules_index_path=Path(
                values.get("RAG_RULING_RULES_INDEX_PATH", str(data_dir / "ruling_rules_chroma"))
            ),
            ruling_faq_collection_name=values.get(
                "RAG_RULING_FAQ_COLLECTION", "ruling_faqs"
            ),
            ruling_rules_collection_name=values.get(
                "RAG_RULING_RULES_COLLECTION", "ocg_rules"
            ),
            ruling_faq_manifest_path=Path(
                values.get(
                    "RAG_RULING_FAQ_MANIFEST",
                    str(data_dir / "ruling_faq_chroma" / "manifest.json"),
                )
            ),
            ruling_rules_manifest_path=Path(
                values.get(
                    "RAG_RULING_RULES_MANIFEST",
                    str(data_dir / "ruling_rules_chroma" / "manifest.json"),
                )
            ),
            ruling_sparse_top_k=int(values.get("RAG_RULING_SPARSE_TOP_K", "20")),
            ruling_dense_top_k=int(values.get("RAG_RULING_DENSE_TOP_K", "20")),
            ruling_fused_top_k=int(values.get("RAG_RULING_FUSED_TOP_K", "20")),
            ruling_final_top_k=int(values.get("RAG_RULING_FINAL_TOP_K", "10")),
            ruling_rerank_provider=values.get("RAG_RULING_RERANK_PROVIDER", "none"),
            riftbound_rag_base_url=values.get(
                "RIFTBOUND_RAG_BASE_URL", "http://127.0.0.1:7862"
            ),
            riftbound_rag_timeout_seconds=_positive_float(
                values.get("RIFTBOUND_RAG_TIMEOUT_SECONDS"), 200.0
            ),
        )


def _positive_float(value: str | None, default: float) -> float:
    """非数字/非正数的环境变量值回退到默认浮点数。"""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
