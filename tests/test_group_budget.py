from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from types import SimpleNamespace

import pytest
from rag_agent.group_budget import GroupBudget,GroupBudgetExceeded,invoke_group_model,CHINA
from rag_agent.config import Settings

NOW=1700000000


def test_atomic_admission_and_restart(tmp_path):
    path=tmp_path/'budget.db'
    def reserve(_):
        try:return GroupBudget(path,daily=.01,monthly=.01).reserve('chat',1000,1000,NOW)
        except GroupBudgetExceeded:return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        tickets=list(pool.map(reserve,range(4)))
    assert sum(t is not None for t in tickets)==1
    assert not reserve(0)
    status=GroupBudget(path).status(NOW)
    assert status['estimated_cny']==.01 and status['uncertain_calls']==1


def test_settle_usage_missing_usage_and_no_double_settle(tmp_path):
    budget=GroupBudget(tmp_path/'budget.db')
    ticket=budget.reserve('learning',10000,1000,NOW)
    result=SimpleNamespace(usage_metadata=dict(input_tokens=2000,output_tokens=100,
        input_token_details=dict(cache_read=500),output_token_details=dict(reasoning=0)))
    budget.settle(ticket,result)
    assert budget.status(NOW)['estimated_cny']==.0048
    assert budget.status(NOW)['cache_tokens']==500
    budget.settle(ticket,SimpleNamespace())
    budget.settle(ticket,SimpleNamespace(usage_metadata=dict(input_tokens=0,output_tokens=0)))
    assert budget.status(NOW)['estimated_cny']==.0048
    missing=budget.reserve('chat',1000,100,NOW)
    budget.settle(missing,SimpleNamespace())
    assert budget.status(NOW)['uncertain_calls']==1
    with closing(budget.connect()) as db:
        columns={r[1] for r in db.execute('pragma table_info(group_model_usage)')}
    assert not columns & {'prompt','messages','response','text'}


def test_daily_and_monthly_beijing_boundaries(tmp_path):
    budget=GroupBudget(tmp_path/'budget.db',daily=.01,monthly=.02)
    t=datetime(2026,10,31,23,59,tzinfo=CHINA).timestamp()
    budget.reserve('chat',1000,1000,t)
    with pytest.raises(GroupBudgetExceeded): budget.reserve('chat',1000,1000,t)
    assert budget.reserve('chat',1000,1000,t+120)
    assert budget.status(t+120)['calls']==1
    # Daily reset does not bypass the monthly budget.
    budget.reserve('chat',1000,1000,t+86400+120)
    with pytest.raises(GroupBudgetExceeded):budget.reserve('chat',1000,1000,t+2*86400+120)


def test_failure_reservation_and_thinking_disabled(tmp_path,monkeypatch):
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))
    captured={}
    class Model:
        def bind(self,**kw):captured.update(kw);return self
        def invoke(self,messages):raise TimeoutError()
    with pytest.raises(TimeoutError):
        invoke_group_model(lambda s:Model(),Settings.from_env({}),'chat',[],max_tokens=100)
    assert captured['extra_body']=={'thinking':{'type':'disabled'}}
    assert GroupBudget().status()['uncertain_calls']==1
    assert GroupBudget().status()['estimated_cny']>0


def test_budget_rejection_prevents_model_creation(tmp_path,monkeypatch):
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))
    monkeypatch.setenv('GROUP_CHAT_MONTHLY_BUDGET_CNY','0')
    def forbidden(settings):raise AssertionError('Model must not be created')
    with pytest.raises(GroupBudgetExceeded):
        invoke_group_model(forbidden,Settings.from_env({}),'chat',[],max_tokens=100)
