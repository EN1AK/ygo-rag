from rag_agent.retrieval.embeddings import BgeM3Embeddings
from rag_agent.retrieval.hybrid import Candidate
from rag_agent.retrieval.reranker import BgeReranker, SubprocessReranker


def test_embedding_adapter_stores_model_name_without_loading_dependency():
    adapter = BgeM3Embeddings("BAAI/bge-m3", device="cuda")

    assert adapter.model_name == "BAAI/bge-m3"
    assert adapter.device == "cuda"
    assert adapter.backend == "auto"


def test_reranker_adapter_stores_model_name_without_loading_dependency():
    reranker = BgeReranker("BAAI/bge-reranker-v2-m3", device="cuda")

    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert reranker.device == "cuda"


def test_subprocess_reranker_parses_worker_json(monkeypatch):
    captured = {}

    class Completed:
        returncode = 0
        stdout = '{"scores":[2.5,1.0]}'
        stderr = ""

    def fake_run(command, *, input, text, capture_output, encoding, errors, timeout, env):
        captured["command"] = command
        captured["input"] = input
        captured["env"] = env
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)
    reranker = SubprocessReranker(
        "BAAI/bge-reranker-v2-m3",
        device="auto",
        python_executable="python",
    )
    candidates = [
        Candidate(1, 0.1, "hybrid", {"name": "A", "desc": "alpha"}),
        Candidate(2, 0.2, "hybrid", {"name": "B", "desc": "beta"}),
    ]

    results = reranker.rerank("query", candidates)

    assert [candidate.card_id for candidate in results] == [1, 2]
    assert [candidate.score for candidate in results] == [2.5, 1.0]
    assert results[0].source == "reranker"
    assert "rag_agent.rerank_worker" in captured["command"]
    assert '"query": "query"' in captured["input"]
    assert captured["env"]["RAG_DEVICE"] == "auto"
