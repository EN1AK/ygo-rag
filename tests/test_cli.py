import sqlite3
import os
import subprocess
import sys


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


def run_cli(*args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
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
    assert "web" in result.stdout


def test_cli_query_help_lists_rag_mode_flags():
    result = run_cli("query", "--help")

    assert result.returncode == 0
    assert "--semantic" in result.stdout
    assert "--rerank" in result.stdout
    assert "--llm-rerank" in result.stdout
    assert "--llm" in result.stdout


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
