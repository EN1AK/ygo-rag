from pathlib import Path
import sqlite3

import pytest

from rag_agent.card_metadata import TYPE_EFFECT, TYPE_MONSTER, TYPE_XYZ
from rag_agent.cards import Card
from rag_agent.config import Settings
from rag_agent.retrieval.hybrid import Candidate
from rag_agent.query_service import (
    QueryRequest,
    build_agent,
    execute_query,
    resolve_rerank_provider,
)


def settings(**overrides):
    values = {
        "RAG_DATA_DIR": "data",
        "DEEPSEEK_API_KEY": "test-key",
    }
    values.update(overrides)
    return Settings.from_env(values)


def cards():
    return [Card(1, "我身作盾", "支付1500基本分，使破坏效果无效并破坏。")]


def test_resolve_rerank_provider_defaults_to_none():
    request = QueryRequest(query="q", db_path=Path("cards.cdb"))

    assert resolve_rerank_provider(request, settings()) == "none"


def test_resolve_rerank_provider_keeps_rerank_true_as_local():
    request = QueryRequest(query="q", db_path=Path("cards.cdb"), rerank=True)

    assert resolve_rerank_provider(request, settings()) == "local"


def test_resolve_rerank_provider_accepts_explicit_llm():
    request = QueryRequest(query="q", db_path=Path("cards.cdb"), rerank_provider="llm")

    assert resolve_rerank_provider(request, settings()) == "llm"


def test_build_agent_configures_llm_reranker_without_final_llm():
    request = QueryRequest(
        query="有没有效果类似“我身作盾”的卡",
        db_path=Path("cards.cdb"),
        rerank_provider="llm",
        use_llm=False,
    )
    agent = build_agent(
        cards(),
        settings(),
        semantic=False,
        rerank=False,
        use_llm=request.use_llm,
        rerank_provider=request.rerank_provider,
    )

    assert agent.reranker is not None
    assert agent.llm is None


def test_build_agent_requires_deepseek_key_for_llm_rerank():
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        build_agent(
            cards(),
            settings(DEEPSEEK_API_KEY=""),
            semantic=False,
            rerank=False,
            use_llm=False,
            rerank_provider="llm",
        )


def test_llm_rerank_failure_warning_is_collected():
    warnings = []

    class FailingLlm:
        def invoke(self, prompt):
            raise TimeoutError("request timed out")

    agent = build_agent(
        cards(),
        settings(),
        semantic=False,
        rerank=False,
        use_llm=False,
        rerank_provider="llm",
        warnings=warnings,
    )
    agent.reranker.llm = FailingLlm()

    reranked = agent.reranker.rerank(
        "query",
        [Candidate(1, 0.1, "hybrid", {"name": "我身作盾", "desc": "效果"})],
    )

    assert [candidate.card_id for candidate in reranked] == [1]
    assert warnings
    assert "falling back to hybrid candidate order" in warnings[0]


def test_execute_query_exposes_structured_filters_and_diagnostics(tmp_path):
    db_path = tmp_path / "cards.cdb"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "create table datas(id integer primary key, ot integer, alias integer, setcode integer, type integer, atk integer, def integer, level integer, race integer, attribute integer, category integer)"
        )
        connection.execute(
            "create table texts(id integer primary key, name text, desc text, str1 text, str2 text, str3 text, str4 text, str5 text, str6 text, str7 text, str8 text, str9 text, str10 text, str11 text, str12 text, str13 text, str14 text, str15 text, str16 text)"
        )
        connection.execute(
            "insert into datas values(1, 0, 0, 0, ?, 1000, 1000, 4, 8192, 32, 0)",
            (TYPE_MONSTER | TYPE_EFFECT | TYPE_XYZ,),
        )
        connection.execute(
            "insert into texts values(1, '四阶超量', '除外对手墓地的卡。', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')"
        )
        connection.execute(
            "insert into datas values(2, 0, 0, 0, ?, 1000, 1000, 4, 8192, 32, 0)",
            (TYPE_MONSTER | TYPE_EFFECT,),
        )
        connection.execute(
            "insert into texts values(2, '四星怪兽', '除外对手墓地的卡。', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')"
        )
        connection.commit()
    finally:
        connection.close()

    response = execute_query(
        QueryRequest(
            query="效果是除外对手墓地的卡的四星超量怪兽",
            db_path=db_path,
            top_k=5,
        ),
        settings(),
    )

    assert [result["card_id"] for result in response.results] == [1]
    assert response.structured_query["has_filters"] is True
    assert response.structured_query["filters"]["rank"] == 4
    assert response.filter_diagnostics["applied"] is True
    assert response.filter_diagnostics["filtered_candidates"] == 1
