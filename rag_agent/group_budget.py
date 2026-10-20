"""Conservative persistent CNY accounting for the single-group model routes."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

CHINA = timezone(timedelta(hours=8))


class GroupBudgetExceeded(RuntimeError):
    pass


def amount(name, default):
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid group budget configuration")
    return value


class GroupBudget:
    def __init__(self, path=None, *, daily=None, monthly=None):
        self.path = Path(path or os.getenv("GROUP_CHAT_BUDGET_DB", "data/group_chat_budget.db"))
        self.daily = amount("GROUP_CHAT_DAILY_BUDGET_CNY", 1) if daily is None else daily
        self.monthly = amount("GROUP_CHAT_MONTHLY_BUDGET_CNY", 10) if monthly is None else monthly
        self.input_rate = amount("GROUP_CHAT_INPUT_CNY_PER_MILLION", 2)
        self.output_rate = amount("GROUP_CHAT_OUTPUT_CNY_PER_MILLION", 8)

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("""CREATE TABLE IF NOT EXISTS group_model_usage (
            id TEXT PRIMARY KEY, day TEXT NOT NULL, route TEXT NOT NULL,
            input_tokens INTEGER, output_tokens INTEGER, cache_tokens INTEGER,
            reasoning_tokens INTEGER, cost REAL NOT NULL, state TEXT NOT NULL,
            created REAL NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS group_model_usage_day ON group_model_usage(day)")
        db.commit()
        return db

    def cost(self, inputs, outputs):
        # Peak uncached prices deliberately overestimate cache/off-peak savings.
        return (inputs*self.input_rate + outputs*self.output_rate)/1_000_000

    @staticmethod
    def day(now):
        return datetime.fromtimestamp(now, CHINA).date().isoformat()

    def reserve(self, route, inputs, outputs, now=None):
        now = time.time() if now is None else now
        day = self.day(now)
        ceiling = self.cost(inputs, outputs)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            daily = db.execute("SELECT COALESCE(SUM(cost),0) FROM group_model_usage WHERE day=?", (day,)).fetchone()[0]
            monthly = db.execute("SELECT COALESCE(SUM(cost),0) FROM group_model_usage WHERE substr(day,1,7)=?", (day[:7],)).fetchone()[0]
            if daily+ceiling > self.daily or monthly+ceiling > self.monthly:
                raise GroupBudgetExceeded("Group model budget exhausted")
            ticket = uuid4().hex
            db.execute("INSERT INTO group_model_usage VALUES (?,?,?,NULL,NULL,NULL,NULL,?,'reserved',?)", (ticket,day,route,ceiling,now))
            # Keep a year of numeric audit data, never chat text.
            db.execute("DELETE FROM group_model_usage WHERE day<?", (self.day(now-366*86400),))
            return ticket

    def settle(self, ticket, result):
        usage = getattr(result, "usage_metadata", None) or {}
        raw = (getattr(result, "response_metadata", None) or {}).get("token_usage", {}) or {}
        inputs = usage.get("input_tokens", raw.get("prompt_tokens"))
        outputs = usage.get("output_tokens", raw.get("completion_tokens"))
        if any(type(v) is not int or v < 0 for v in (inputs, outputs)):
            return  # Unknown billing remains charged at the reserved upper estimate.
        cache = (usage.get("input_token_details") or {}).get("cache_read", raw.get("prompt_cache_hit_tokens", 0))
        reasoning = (usage.get("output_token_details") or {}).get("reasoning", (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        cache = cache if type(cache) is int and cache >= 0 else 0
        reasoning = reasoning if type(reasoning) is int and reasoning >= 0 else 0
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE group_model_usage SET input_tokens=?,output_tokens=?,
                cache_tokens=?,reasoning_tokens=?,cost=?,state='metered'
                WHERE id=? AND state='reserved'""", (inputs,outputs,cache,reasoning,self.cost(inputs,outputs),ticket))

    def status(self, now=None):
        day = self.day(time.time() if now is None else now)
        with closing(self.connect()) as db:
            row = db.execute("""SELECT COUNT(*),COALESCE(SUM(cost),0),
                COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),
                COALESCE(SUM(cache_tokens),0),COALESCE(SUM(reasoning_tokens),0),
                COALESCE(SUM(state='reserved'),0) FROM group_model_usage WHERE substr(day,1,7)=?""", (day[:7],)).fetchone()
            today = db.execute("SELECT COALESCE(SUM(cost),0) FROM group_model_usage WHERE day=?", (day,)).fetchone()[0]
        return dict(month=day[:7], calls=row[0], estimated_cny=round(row[1],6),
                    input_tokens=row[2], output_tokens=row[3], cache_tokens=row[4],
                    reasoning_tokens=row[5], uncertain_calls=row[6],
                    daily_estimated_cny=round(today,6), daily_limit_cny=self.daily,
                    monthly_limit_cny=self.monthly)


def invoke_group_model(factory, settings, route, messages, **options):
    budget = GroupBudget()
    # UTF-8 bytes bound text tokens conservatively; allowance covers role framing.
    inputs = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))+1024
    ticket = budget.reserve(route, inputs, options["max_tokens"])
    result = factory(settings).bind(
        **options, extra_body={"thinking": {"type": "disabled"}},
    ).invoke(messages)
    # Settle before output validation: malformed model responses still cost money.
    budget.settle(ticket, result)
    return result
