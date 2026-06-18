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
    assert settings.deepseek_api_key is None


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
