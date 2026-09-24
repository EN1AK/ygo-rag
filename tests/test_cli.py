import sqlite3
import json
import os
import subprocess
import sys

import rag_agent.__main__ as cli
from rag_agent.assistant_agent import AssistantAgentResponse
from rag_agent.adjudication import (
    AdjudicationProgressEvent,
    AdjudicationResponse,
    ParsedScenario,
    SimulationStatus,
)
from rag_agent.adjudication_service import render_adjudication_answer
from rag_agent.card_metadata import TYPE_EFFECT, TYPE_MONSTER, TYPE_XYZ
from rag_agent.translation_service import (
    TranslationResponse,
    build_structured_translation_response,
)


def create_mini_cards_db(path):
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "create table datas(id integer primary key, ot integer, alias integer, setcode integer, type integer, atk integer, def integer, level integer, race integer, attribute integer, category integer)"
        )
        connection.execute(
            "create table texts(id integer primary key, name text, desc text, str1 text, str2 text, str3 text, str4 text, str5 text, str6 text, str7 text, str8 text, str9 text, str10 text, str11 text, str12 text, str13 text, str14 text, str15 text, str16 text)"
        )
        connection.execute(
            "insert into datas values(1, 0, 0, 0, 2, -2, -2, 0, 0, 0, 0)"
        )
        connection.execute(
            "insert into texts values(1, '我身作盾', '支付1500基本分，使破坏怪兽的效果发动无效并破坏。', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')"
        )
        connection.commit()
    finally:
        connection.close()


def run_cli(*args, env_overrides=None):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "rag_agent", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


def test_cli_help_lists_commands():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "download-db" in result.stdout
    assert "inspect-db" in result.stdout
    assert "download-rulings" in result.stdout
    assert "build-ruling-index" in result.stdout
    assert "inspect-ruling-index" in result.stdout
    assert "query-rulings" in result.stdout
    assert "build-index" in result.stdout
    assert "query" in result.stdout
    assert "translate" in result.stdout
    assert "adjudicate" in result.stdout
    assert "agent" in result.stdout
    assert "rift-query" in result.stdout
    assert "web" in result.stdout


def test_cli_query_help_lists_rag_mode_flags():
    result = run_cli("query", "--help")

    assert result.returncode == 0
    assert "--semantic" in result.stdout
    assert "--rerank" in result.stdout
    assert "--llm-rerank" in result.stdout
    assert "--llm" in result.stdout


def test_cli_translate_help_lists_language_options():
    result = run_cli("translate", "--help")

    assert result.returncode == 0
    assert "--source-lang" in result.stdout
    assert "--target-lang" in result.stdout
    assert "--structured-max-block-chars" in result.stdout


def test_cli_adjudicate_help_lists_evidence_options():
    result = run_cli("adjudicate", "--help")

    assert result.returncode == 0
    assert "--ygocdb-cache" in result.stdout
    assert "--rules-epub" in result.stdout
    assert "--simulate" in result.stdout
    assert "--agent" in result.stdout
    assert "--trace" in result.stdout
    assert "--progress" in result.stdout
    assert "--output-format" in result.stdout


def test_cli_agent_help_lists_unified_tool_options():
    result = run_cli("agent", "--help")

    assert result.returncode == 0
    assert "--semantic" in result.stdout
    assert "--llm-rerank" in result.stdout
    assert "--max-steps" in result.stdout
    assert "--trace" in result.stdout
    assert "--progress" in result.stdout


def test_cli_rift_query_help_lists_options():
    result = run_cli("rift-query", "--help")

    assert result.returncode == 0
    assert "--top-k" in result.stdout
    assert "--mode" in result.stdout
    assert "--trace" in result.stdout


def test_cli_rift_query_success_output(monkeypatch, capsys):
    captured = {}

    def fake_execute(request, settings, transport=None):
        captured["request"] = request
        return {
            "answer": "连锁结算无次数上限",
            "warnings": ["部分证据未覆盖"],
            "sources": [
                {"rule_id": "R-CR-716.1", "topic": "连锁"},
                {"rule_id": "R-CR-716.10", "topic": "连锁"},
            ],
        }

    monkeypatch.setattr("rag_agent.rift_gateway.execute_rift_query", fake_execute)

    exit_code = cli.main(
        ["rift-query", "反制堆叠有上限吗", "--top-k", "4",
         "--mode", "agent", "--trace"]
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert "连锁结算无次数上限" in output.out
    assert "R-CR-716.1" in output.out and "R-CR-716.10" in output.out
    assert "warning: 部分证据未覆盖" in output.err
    request = captured["request"]
    assert request.top_k == 4
    assert request.mode == "agent"
    assert request.trace is True


def test_cli_rift_query_defaults_omit_mode_and_trace(monkeypatch):
    captured = {}

    def fake_execute(request, settings, transport=None):
        captured["request"] = request
        return {"answer": "a", "warnings": [], "sources": []}

    monkeypatch.setattr("rag_agent.rift_gateway.execute_rift_query", fake_execute)

    assert cli.main(["rift-query", "什么是连锁"]) == 0
    request = captured["request"]
    assert request.top_k == 6
    assert request.mode is None
    assert request.trace is None


def test_cli_rift_query_gateway_failure_exits_1(monkeypatch, capsys):
    from rag_agent.rift_gateway import RiftGatewayError

    def fake_execute(request, settings, transport=None):
        raise RiftGatewayError("unavailable", "upstream unavailable: refused")

    monkeypatch.setattr("rag_agent.rift_gateway.execute_rift_query", fake_execute)

    assert cli.main(["rift-query", "什么是连锁"]) == 1
    assert "error: upstream unavailable: refused" in capsys.readouterr().err


def test_cli_agent_uses_unified_service(monkeypatch, capsys):
    seen = {}

    def fake_execute(request, settings, *, progress_callback=None):
        seen["request"] = request
        return AssistantAgentResponse(
            answer="统一 Agent 回答",
            tool_results=(),
            warnings=(),
            trace=(),
        )

    monkeypatch.setattr(cli, "execute_assistant_agent", fake_execute)

    result = cli.main(
        [
            "agent",
            "查找我身作盾",
            "--semantic",
            "--max-steps",
            "4",
            "--output-format",
            "natural",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out.strip() == "统一 Agent 回答"
    assert seen["request"].text == "查找我身作盾"
    assert seen["request"].semantic is True
    assert seen["request"].max_steps == 4


def test_cli_download_rulings_help_lists_cache_options():
    result = run_cli("download-rulings", "--help")

    assert result.returncode == 0
    assert "--out" in result.stdout
    assert "--limit" in result.stdout
    assert "--delay" in result.stdout
    assert "--force" in result.stdout
    assert "--use-proxy" in result.stdout


def test_cli_build_ruling_index_help_lists_modes():
    result = run_cli("build-ruling-index", "--help")

    assert result.returncode == 0
    assert "--faq-only" in result.stdout
    assert "--rules-only" in result.stdout
    assert "--reset" in result.stdout
    assert "--batch-size" in result.stdout


def test_cli_inspect_ruling_index_reports_missing_indexes():
    result = run_cli(
        "inspect-ruling-index",
        "--json",
        env_overrides={
            "RAG_DATA_DIR": "data/missing-ruling-index-test",
        },
    )

    assert result.returncode == 0
    assert "faq" in result.stdout
    assert "rules" in result.stdout
    assert "usable" in result.stdout


def test_cli_query_rulings_returns_evidence_shape(tmp_path):
    db_path = tmp_path / "cards.cdb"
    cache_path = tmp_path / "ygocdb"
    cache_path.mkdir()
    rules_path = tmp_path / "missing.epub"
    create_mini_cards_db(db_path)
    (cache_path / "1.json").write_text(
        json.dumps(
            {
                "id": 1,
                "faqs": [
                    {
                        "fid": "q1",
                        "title": "Shield ruling",
                        "question": "Can it negate destruction?",
                        "answer": "It can negate and destroy.",
                        "refer": {},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "query-rulings",
        "negate destruction",
        "--db",
        str(db_path),
        "--ygocdb-cache",
        str(cache_path),
        "--rules-epub",
        str(rules_path),
        "--json",
    )

    assert result.returncode == 0
    assert "ygocdb:faq:q1" in result.stdout
    assert "candidates" in result.stdout


def test_cli_translate_defaults_to_chinese(monkeypatch, capsys):
    seen = {}

    def fake_execute_translation(request, settings):
        seen["request"] = request
        return TranslationResponse(
            translation="你好",
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            warnings=[],
            structured=build_structured_translation_response(
                "你好",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            ),
        )

    monkeypatch.setattr(cli, "execute_translation", fake_execute_translation)

    result = cli.main(["translate", "hello"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out.strip() == "你好"
    assert seen["request"].text == "hello"
    assert seen["request"].source_lang == "auto"
    assert seen["request"].target_lang == "zh-CN"


def test_cli_translate_accepts_explicit_target_language(monkeypatch, capsys):
    seen = {}

    def fake_execute_translation(request, settings):
        seen["request"] = request
        return TranslationResponse(
            translation="hello",
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            warnings=[],
            structured=build_structured_translation_response(
                "hello",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            ),
        )

    monkeypatch.setattr(cli, "execute_translation", fake_execute_translation)

    result = cli.main(["translate", "你好", "--target-lang", "en"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out.strip() == "hello"
    assert seen["request"].source_lang == "auto"
    assert seen["request"].target_lang == "en"


def test_cli_translate_empty_input_uses_error_handling(capsys):
    result = cli.main(["translate", " "])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.err.startswith("error:")
    assert "text is required" in captured.err


def test_cli_inspect_db_reports_counts(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli("inspect-db", "--db", str(db_path))

    assert result.returncode == 0
    assert "datas: 1" in result.stdout
    assert "texts: 1" in result.stdout


def test_download_rulings_saves_payload_and_skips_existing(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "cards.cdb"
    cache_path = tmp_path / "ygocdb"
    create_mini_cards_db(db_path)
    calls = []

    def fake_fetch(self, card_id, *, timeout=30.0, use_proxy=True):
        calls.append(card_id)
        assert use_proxy is False
        return {"id": card_id, "faqs": [{"fid": "1"}], "supplement": None}

    monkeypatch.setattr(cli.YgocdbCache, "fetch_card_payload", fake_fetch)

    result = cli.download_rulings(
        db_path,
        cache_path,
        delay_seconds=0,
        timeout_seconds=1,
    )
    result_again = cli.download_rulings(
        db_path,
        cache_path,
        delay_seconds=0,
        timeout_seconds=1,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert result_again == 0
    assert calls == [1]
    assert (cache_path / "1.json").exists()
    assert "downloaded=1" in captured.out
    assert "skipped=1" in captured.out


def test_cli_query_uses_sparse_baseline_when_db_is_provided(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli("query", "有没有效果类似“我身作盾”的卡", "--db", str(db_path))

    assert result.returncode == 0
    assert "我身作盾" in result.stdout
    assert "原文：" in result.stdout


def test_cli_adjudicate_runs_without_optional_evidence(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli(
        "adjudicate",
        "Can 我身作盾 apply?",
        "--db",
        str(db_path),
        "--ygocdb-cache",
        str(tmp_path / "missing-ygocdb"),
        "--rules-epub",
        str(tmp_path / "missing.epub"),
    )

    assert result.returncode == 0
    assert "warning:" in result.stderr
    assert "scenario" in result.stdout


def test_cli_adjudicate_agent_outputs_coverage(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli(
        "adjudicate",
        "Can 我身作盾 apply?",
        "--db",
        str(db_path),
        "--ygocdb-cache",
        str(tmp_path / "missing-ygocdb"),
        "--rules-epub",
        str(tmp_path / "missing.epub"),
        "--agent",
        "--trace",
    )

    assert result.returncode == 0
    assert "agent_coverage" in result.stdout
    assert "agent_trace" in result.stdout


def test_cli_adjudicate_json_output_is_machine_readable(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli(
        "adjudicate",
        "Can 我身作盾 apply?",
        "--db",
        str(db_path),
        "--ygocdb-cache",
        str(tmp_path / "missing-ygocdb"),
        "--rules-epub",
        str(tmp_path / "missing.epub"),
        "--output-format",
        "json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["scenario"]["original_text"] == "Can 我身作盾 apply?"


def test_render_adjudication_answer_produces_natural_chinese():
    answer = json.dumps(
        {
            "conclusion": "可以适用。",
            "source_status": {
                "faq": "found",
                "rules": "not_found",
                "failure_reason": "已依据 FAQ 证据完成裁定，但未找到对应的规则文档机制。",
            },
            "answers": [
                {
                    "question": "该效果可以适用吗？",
                    "case_conclusion": "可以。",
                    "hypothetical_conclusion": "",
                    "reasoning": [],
                }
            ],
            "evidence_use": [],
            "uncertainties": [],
        },
        ensure_ascii=False,
    )

    rendered = render_adjudication_answer(answer)

    assert "裁定：可以适用。" in rendered
    assert "FAQ 已找到；规则文档 未找到" in rendered
    assert "未找到对应的规则文档机制" in rendered
    assert '"source_status"' not in rendered


def test_cli_progress_callback_prints_public_event_message_to_stderr(monkeypatch, capsys, tmp_path):
    def fake_execute(request, settings):
        assert request.progress_callback is not None
        request.progress_callback(
            AdjudicationProgressEvent(stage="parse", message="场景解析完成。")
        )
        return AdjudicationResponse(
            scenario=ParsedScenario(original_text=request.scenario),
            resolved_cards=(),
            evidence=(),
            reasoning=(),
            answer="暂无可靠裁定。",
            confidence="low",
            warnings=(),
            simulation=SimulationStatus(status="not_requested"),
        )

    monkeypatch.setattr(cli, "execute_adjudication", fake_execute)

    result = cli.adjudicate_scenario(
        "测试场景",
        tmp_path / "cards.cdb",
        cli.Settings.from_env({"RAG_DATA_DIR": str(tmp_path)}),
        show_progress=True,
        output_format="natural",
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out.strip() == "暂无可靠裁定。"
    assert captured.err.strip() == "场景解析完成。"


def test_cli_llm_rerank_reports_progress_before_missing_key_error(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli(
        "query",
        "有没有效果类似“我身作盾”的卡",
        "--db",
        str(db_path),
        "--llm-rerank",
        env_overrides={"DEEPSEEK_API_KEY": ""},
    )

    assert result.returncode == 1
    assert "Calling DeepSeek LLM judge rerank" in result.stderr
    assert "DEEPSEEK_API_KEY is required" in result.stderr


def test_cli_query_prints_structured_filter_diagnostics(tmp_path):
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
        connection.commit()
    finally:
        connection.close()

    result = run_cli("query", "效果是除外对手墓地的卡的四星超量怪兽", "--db", str(db_path))

    assert result.returncode == 0
    assert "Structured filters:" in result.stdout
    assert '"rank": 4' in result.stdout
    assert "四阶超量" in result.stdout
