from rag_agent.config import Settings


def test_settings_defaults_keep_runtime_artifacts_under_data():
    settings = Settings.from_env({})

    assert settings.chroma_persist_dir.as_posix().endswith("data/chroma")
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.device == "auto"
    assert settings.embedding_device == "auto"
    assert settings.reranker_device == "auto"
    assert settings.reranker_device_explicit is False
    assert settings.rerank_provider == "none"
    assert settings.llm_rerank_max_candidates == 20
    assert settings.deepseek_timeout_seconds == 60.0
    assert settings.deepseek_api_key is None
    assert settings.riftbound_rag_base_url == "http://127.0.0.1:7862"
    assert settings.riftbound_rag_timeout_seconds == 200.0
    assert settings.ruling_faq_index_path.as_posix().endswith("data/ruling_faq_chroma")
    assert settings.ruling_rules_index_path.as_posix().endswith("data/ruling_rules_chroma")
    assert settings.ruling_faq_collection_name == "ruling_faqs"
    assert settings.ruling_rules_collection_name == "ocg_rules"
    assert settings.ruling_sparse_top_k == 20
    assert settings.ruling_dense_top_k == 20
    assert settings.ruling_fused_top_k == 20
    assert settings.ruling_final_top_k == 10
    assert settings.ruling_rerank_provider == "none"
    assert settings.adjudication_agent_max_iterations == 8
    assert settings.adjudication_agent_total_turn_limit == 14
    assert settings.adjudication_agent_retrieval_limit == 8
    assert settings.adjudication_agent_critique_limit == 5
    assert settings.adjudication_agent_submit_limit == 2
    assert settings.adjudication_card_resolution_top_k == 5
    assert settings.adjudication_card_resolution_retry_limit == 1
    assert settings.adjudication_llm_scenario_parsing is True
    assert settings.adjudication_llm_card_extraction is True
    assert settings.adjudication_llm_card_selection is False


def test_settings_reads_device_from_env():
    settings = Settings.from_env({"RAG_DEVICE": "cuda"})

    assert settings.device == "cuda"
    assert settings.embedding_device == "cuda"
    assert settings.reranker_device == "cuda"
    assert settings.reranker_device_explicit is False


def test_settings_allows_separate_embedding_and_reranker_devices():
    settings = Settings.from_env(
        {
            "RAG_DEVICE": "cuda",
            "RAG_EMBEDDING_DEVICE": "cuda",
            "RAG_RERANKER_DEVICE": "auto",
        }
    )

    assert settings.embedding_device == "cuda"
    assert settings.reranker_device == "auto"
    assert settings.reranker_device_explicit is True


def test_settings_reads_deepseek_timeout_from_env():
    settings = Settings.from_env({"DEEPSEEK_TIMEOUT_SECONDS": "15"})

    assert settings.deepseek_timeout_seconds == 15.0


def test_settings_reads_ruling_index_env_values():
    settings = Settings.from_env(
        {
            "RAG_DATA_DIR": "data",
            "RAG_RULING_FAQ_INDEX_PATH": "custom/faq",
            "RAG_RULING_RULES_INDEX_PATH": "custom/rules",
            "RAG_RULING_FAQ_COLLECTION": "faq_collection",
            "RAG_RULING_RULES_COLLECTION": "rules_collection",
            "RAG_RULING_SPARSE_TOP_K": "11",
            "RAG_RULING_DENSE_TOP_K": "12",
            "RAG_RULING_FUSED_TOP_K": "13",
            "RAG_RULING_FINAL_TOP_K": "14",
            "RAG_RULING_RERANK_PROVIDER": "llm",
        }
    )

    assert settings.ruling_faq_index_path.as_posix() == "custom/faq"
    assert settings.ruling_rules_index_path.as_posix() == "custom/rules"
    assert settings.ruling_faq_collection_name == "faq_collection"
    assert settings.ruling_rules_collection_name == "rules_collection"
    assert settings.ruling_sparse_top_k == 11
    assert settings.ruling_dense_top_k == 12
    assert settings.ruling_fused_top_k == 13
    assert settings.ruling_final_top_k == 14
    assert settings.ruling_rerank_provider == "llm"


def test_settings_reads_card_resolution_loop_env_values():
    settings = Settings.from_env(
        {
            "RAG_ADJUDICATION_CARD_RESOLUTION_TOP_K": "7",
            "RAG_ADJUDICATION_CARD_RESOLUTION_RETRY_LIMIT": "3",
            "RAG_ADJUDICATION_LLM_SCENARIO_PARSING": "0",
            "RAG_ADJUDICATION_LLM_CARD_EXTRACTION": "0",
            "RAG_ADJUDICATION_LLM_CARD_SELECTION": "1",
        }
    )

    assert settings.adjudication_card_resolution_top_k == 7
    assert settings.adjudication_card_resolution_retry_limit == 3
    assert settings.adjudication_llm_scenario_parsing is False
    assert settings.adjudication_llm_card_extraction is False
    assert settings.adjudication_llm_card_selection is True


def test_settings_reads_adjudication_agent_budget_env_values():
    settings = Settings.from_env(
        {
            "RAG_ADJUDICATION_AGENT_MAX_ITERATIONS": "6",
            "RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT": "9",
            "RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT": "4",
            "RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT": "3",
            "RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT": "2",
        }
    )

    assert settings.adjudication_agent_max_iterations == 6
    assert settings.adjudication_agent_total_turn_limit == 9
    assert settings.adjudication_agent_retrieval_limit == 4
    assert settings.adjudication_agent_critique_limit == 3
    assert settings.adjudication_agent_submit_limit == 2


def test_settings_reads_riftbound_gateway_env_values():
    settings = Settings.from_env(
        {
            "RIFTBOUND_RAG_BASE_URL": "http://192.168.0.3:7862",
            "RIFTBOUND_RAG_TIMEOUT_SECONDS": "120.5",
        }
    )

    assert settings.riftbound_rag_base_url == "http://192.168.0.3:7862"
    assert settings.riftbound_rag_timeout_seconds == 120.5


def test_settings_invalid_riftbound_timeout_falls_back_to_default():
    for bad in ("abc", "0", "-5", ""):
        settings = Settings.from_env({"RIFTBOUND_RAG_TIMEOUT_SECONDS": bad})

        assert settings.riftbound_rag_timeout_seconds == 200.0
