import sqlite3

from rag_agent.db import inspect_cards_db, load_cards


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
            "insert into texts values(1, '我身作盾', '当对方发动破坏场上怪兽的效果时，可以支付1500基本分使其发动无效并破坏。', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')"
        )
        connection.commit()
    finally:
        connection.close()


def test_load_cards_reads_datas_and_texts(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    cards = load_cards(db_path)

    assert len(cards) == 1
    assert cards[0].card_id == 1
    assert cards[0].name == "我身作盾"
    assert "发动无效并破坏" in cards[0].description
    assert cards[0].type == 2


def test_inspect_cards_db_reports_schema_and_counts(tmp_path):
    db_path = tmp_path / "cards.cdb"
    create_mini_cards_db(db_path)

    info = inspect_cards_db(db_path)

    assert info.path == db_path
    assert info.table_counts["datas"] == 1
    assert info.table_counts["texts"] == 1
    assert "id" in info.columns["texts"]

