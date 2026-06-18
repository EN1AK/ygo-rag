from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rag_agent.cards import Card


@dataclass(frozen=True)
class CardsDbInfo:
    path: Path
    table_counts: dict[str, int]
    columns: dict[str, list[str]]


def _connect(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"cards.cdb not found: {db_path}")
    return sqlite3.connect(db_path)


def inspect_cards_db(path: Path | str) -> CardsDbInfo:
    db_path = Path(path)
    with _connect(db_path) as connection:
        tables = ["datas", "texts"]
        table_counts: dict[str, int] = {}
        columns: dict[str, list[str]] = {}
        for table in tables:
            rows = connection.execute(
                "select name from sqlite_master where type='table' and name=?", (table,)
            ).fetchall()
            if not rows:
                raise ValueError(f"Expected table missing from cards.cdb: {table}")
            table_counts[table] = int(
                connection.execute(f"select count(*) from {table}").fetchone()[0]
            )
            columns[table] = [
                str(row[1]) for row in connection.execute(f"pragma table_info({table})")
            ]
    return CardsDbInfo(path=db_path, table_counts=table_counts, columns=columns)


def load_cards(path: Path | str) -> list[Card]:
    db_path = Path(path)
    inspect_cards_db(db_path)
    query = """
        select
            texts.id,
            texts.name,
            texts.desc,
            datas.type,
            datas.race,
            datas.attribute,
            datas.atk,
            datas.def,
            datas.level,
            datas.category
        from texts
        left join datas on datas.id = texts.id
        where texts.name is not null and texts.name != ''
        order by texts.id
    """
    with _connect(db_path) as connection:
        rows = connection.execute(query).fetchall()
    return [
        Card(
            card_id=int(row[0]),
            name=str(row[1]),
            description=str(row[2] or ""),
            type=row[3],
            race=row[4],
            attribute=row[5],
            atk=row[6],
            defense=row[7],
            level=row[8],
            category=row[9],
        )
        for row in rows
    ]

