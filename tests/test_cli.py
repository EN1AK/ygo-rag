import sqlite3
import os
import subprocess
import sys

import rag_agent.__main__ as cli
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
    assert "build-index" in result.stdout
    assert "query" in result.stdout
    assert "translate" in result.stdout
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


def test_cli_query_uses_sparse_baseline_when_db_is_provided(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    result = run_cli("query", "有没有效果类似“我身作盾”的卡", "--db", str(db_path))

    assert result.returncode == 0
    assert "我身作盾" in result.stdout
    assert "原文：" in result.stdout


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
